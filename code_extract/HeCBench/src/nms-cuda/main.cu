/* 
 * NMS Benchmarking Framework
 *
 * "Work-Efficient Parallel Non-Maximum Suppression Kernels"
 * Copyright (c) 2019 David Oro et al.
 * 
 * This program is free software: you can redistribute it and/or modify  
 * it under the terms of the GNU General Public License as published by  
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but 
 * WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License 
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda.h>

#define MAX_DETECTIONS  4096
#define N_PARTITIONS    32

void print_help()
{
  printf("\nUsage: nmstest  <detections.txt>  <output.txt>\n\n");
  printf("               detections.txt -> Input file containing the coordinates, width, and scores of detected objects\n");
  printf("               output.txt     -> Output file after performing NMS\n");
  printf("               repeat         -> Kernel execution count\n\n");
}

/* NMS Map kernel */
__global__
void generate_nms_bitmap(const float4* rects, unsigned char* nmsbitmap, const float othreshold)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if(rects[i].w < rects[j].w)
  {
    float area = (rects[j].z + 1.0f) * (rects[j].z + 1.0f);
    float w = fmaxf(0.0f, fminf(rects[i].x + rects[i].z, rects[j].x + rects[j].z) - fmaxf(rects[i].x, rects[j].x) + 1.0f);
    float h = fmaxf(0.0f, fminf(rects[i].y + rects[i].z, rects[j].y + rects[j].z) - fmaxf(rects[i].y, rects[j].y) + 1.0f);
    nmsbitmap[i * MAX_DETECTIONS + j] = (((w * h) / area) < othreshold) && (rects[j].z != 0);
  } 
}

/* NMS Reduce kernel */
__device__ __inline__
void compute_nms_point_mask(unsigned char* pointsbitmap, int cond, int idx, int ndetections)
{
  *pointsbitmap = __syncthreads_and(cond);
}

__global__ void reduce_nms_bitmap(unsigned char* nmsbitmap, unsigned char* pointsbitmap, int ndetections)
{
  int idx = blockIdx.x * MAX_DETECTIONS + threadIdx.x;

  compute_nms_point_mask(&pointsbitmap[blockIdx.x], nmsbitmap[idx], idx, ndetections);

  for(int i=0; i<(N_PARTITIONS-1); i++)
  {
    idx += MAX_DETECTIONS / N_PARTITIONS;
    compute_nms_point_mask(&pointsbitmap[blockIdx.x], pointsbitmap[blockIdx.x] && nmsbitmap[idx], idx, ndetections);
  }
} 

/* Gets the optimal X or Y dimension for a given thread block */
int get_optimal_dim(int val)
{
  int div, neg, cntneg, cntpos;

  /* We start figuring out if 'val' is divisible by 16 
     (e.g. optimal 16x16 thread block of maximum GPU occupancy */

  neg = 1;
  div = 16;
  cntneg = div;
  cntpos = div;

  /* In order to guarantee the ending of this loop if 'val' is 
     a prime number, we limit the loop to 5 iterations */

  for(int i=0; i<5; i++)
  {
    if(val % div == 0)
      return div;

    if(neg)
    {
      cntneg--;
      div = cntneg;
      neg = 0;
    }
    else
    {
      cntpos++;
      div = cntpos;
      neg = 1;
    }
  }

  return 16;
}


/* Gets an upper limit for 'val' multiple of the 'mul' integer */
int get_upper_limit(int val, int mul)
{
  int cnt = mul;

  /* The upper limit must be lower than
     the maximum allowed number of detections */

  while(cnt < val)
    cnt += mul;

  if(cnt > MAX_DETECTIONS)
    cnt = MAX_DETECTIONS;

  return cnt;
}

int main(int argc, char *argv[])
{
  int x, y, w;
  float score;

  if(argc != 4)
  {
    print_help();
    return 0;
  }

  /* Read input detection coordinates from the text file */
  int ndetections = 0;

  FILE *fp = fopen(argv[1], "r");
  if (!fp)
  {
    printf("Error: Unable to open file %s for input detection coordinates.\n", argv[1]);
    return -1;
  }

  /* Memory allocation in the host memory address space */
  float4* cpu_points = (float4*) malloc(sizeof(float4) * MAX_DETECTIONS);
  if(!cpu_points)
  {
    printf("Error: Unable to allocate CPU memory.\n");
    return -1;
  }

  memset(cpu_points, 0, sizeof(float4) * MAX_DETECTIONS);

  while(!feof(fp))
  {
     int cnt = fscanf(fp, "%d,%d,%d,%f\n", &x, &y, &w, &score);

     if (cnt !=4)
     {
	printf("Error: Invalid file format in line %d when reading %s\n", ndetections, argv[1]);
        return -1;
     }
 
    cpu_points[ndetections].x = (float) x;       // x coordinate
    cpu_points[ndetections].y = (float) y;       // y coordinate
    cpu_points[ndetections].z = (float) w;       // window dimensions
    cpu_points[ndetections].w = score;           // score

    ndetections++;
  }

  printf("Number of detections read from input file (%s): %d\n", argv[1], ndetections);

  fclose(fp);

  /* CPU array for storing the detection bitmap */
  unsigned char* cpu_pointsbitmap;
  cpu_pointsbitmap = (unsigned char*) malloc(sizeof(unsigned char) * MAX_DETECTIONS);
  memset(cpu_pointsbitmap, 0, sizeof(unsigned char) * MAX_DETECTIONS);

  /* GPU array for storing the coordinates, dimensions and score of each detected object */

  float4* points;
  cudaMalloc((void**) &points, sizeof(float4) * MAX_DETECTIONS);
  cudaMemset(points, 0, sizeof(float4) * MAX_DETECTIONS);

  /* GPU array for storing the non-maximum supression bitmap */
  unsigned char* nmsbitmap;
  cudaMalloc((void**) &nmsbitmap, sizeof(unsigned char) * MAX_DETECTIONS * MAX_DETECTIONS);
  cudaMemset(nmsbitmap, 1, sizeof(unsigned char) * MAX_DETECTIONS * MAX_DETECTIONS);

  /* GPU array for storing the detection bitmap */
  unsigned char* pointsbitmap;
  cudaMalloc((void**) &pointsbitmap, sizeof(unsigned char) * MAX_DETECTIONS);
  cudaMemset(pointsbitmap, 0, sizeof(unsigned char) * MAX_DETECTIONS);

  /* Transfer detection coordinates read from the input text file to the GPU */
  cudaMemcpy(points, cpu_points, sizeof(float4) * MAX_DETECTIONS, cudaMemcpyHostToDevice);

  /* Execute NMS on the GPU */
  int repeat = atoi(argv[3]);
  int limit = get_upper_limit(ndetections, 16);
  int pkthreads_x = get_optimal_dim(limit);
  int pkthreads_y = get_optimal_dim(limit);
  int pkgrid_x = limit / pkthreads_x;
  int pkgrid_y = limit / pkthreads_y;

  dim3 pkthreads(pkthreads_x, pkthreads_y, 1);
  dim3 pkgrid(pkgrid_x, pkgrid_y, 1);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  /* We build up the non-maximum supression bitmap matrix by removing overlapping windows */
  for (int n = 0; n < repeat; n++)
    generate_nms_bitmap<<<pkgrid, pkthreads>>>(points, nmsbitmap, 0.3f);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (generate_nms_bitmap): %f (s)\n", (time * 1e-9f) / repeat);

  pkthreads.x = MAX_DETECTIONS / N_PARTITIONS; 
  pkthreads.y = 1;
  pkgrid.x = ndetections;
  pkgrid.y = 1;

  start = std::chrono::steady_clock::now();

  /* Then we perform a reduction for generating a point bitmap vector */
  for (int n = 0; n < repeat; n++)
    reduce_nms_bitmap<<<pkgrid, pkthreads>>>(nmsbitmap, pointsbitmap, ndetections);

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (reduce_nms_bitmap): %f (s)\n", (time * 1e-9f) / repeat);

  /* Dump detections after having performed the NMS */

  cudaMemcpy(cpu_pointsbitmap, pointsbitmap, sizeof(unsigned char) * MAX_DETECTIONS, cudaMemcpyDeviceToHost);

  fp = fopen(argv[2], "w");
  if (!fp)
  {
    printf("Error: Unable to open file %s for detection outcome.\n", argv[2]);
    return -1;
  }

  int totaldets = 0;
  for(int i = 0; i < ndetections; i++)
  {
    if(cpu_pointsbitmap[i])
    {
      x = (int) cpu_points[i].x;          // x coordinate
      y = (int) cpu_points[i].y;          // y coordinate
      w = (int) cpu_points[i].z;          // window dimensions
      score = cpu_points[i].w;            // score
      fprintf(fp, "%d,%d,%d,%f\n", x, y, w, score);
      totaldets++; 
    }
  }
  fclose(fp);
  printf("Detections after NMS: %d\n", totaldets);

  cudaFree(points);
  cudaFree(nmsbitmap);
  cudaFree(pointsbitmap);
  free(cpu_points);
  free(cpu_pointsbitmap);

  return 0;
}
