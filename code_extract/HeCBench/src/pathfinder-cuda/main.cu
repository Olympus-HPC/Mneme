/***********************************************************************
 * PathFinder uses dynamic programming to find a path on a 2-D grid from
 * the bottom row to the top row with the smallest accumulated weights,
 * where each step of the path moves straight ahead or diagonally ahead.
 * It iterates row by row, each node picks a neighboring node in the
 * previous row that has the smallest accumulated weight, and adds its
 * own weight to the sum.
 *
 * This kernel uses the technique of ghost zone optimization
 ***********************************************************************/

// Other header files.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <iostream>
#include <sys/time.h>
#include <cuda.h>


// halo width along one direction when advancing to the next iteration
#define HALO     1
#define STR_SIZE 256
#define DEVICE   0
#define M_SEED   9
#define IN_RANGE(x, min, max)  ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))


void fatal(char *s)
{
  fprintf(stderr, "error: %s\n", s);
}

double get_time() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

__global__ void pathfinder (
    const int*__restrict__ gpuWall,
    const int*__restrict__ gpuSrc,
          int*__restrict__ gpuResult,
          int*__restrict__ outputBuffer,
    const int iteration,
    const int theHalo,
    const int borderCols,
    const int cols,
    const int t)
{
  int BLOCK_SIZE = blockDim.x;
  int bx = blockIdx.x;
  int tx = threadIdx.x;
  __shared__ int prev[250];
  __shared__ int result[250];

  // Each block finally computes result for a small block
  // after N iterations.
  // it is the non-overlapping small blocks that cover
  // all the input data

  // calculate the small block size.
  int small_block_cols = BLOCK_SIZE - (iteration*theHalo*2);

  // calculate the boundary for the block according to
  // the boundary of its small block
  int blkX = (small_block_cols*bx) - borderCols;
  int blkXmax = blkX+BLOCK_SIZE-1;

  // calculate the global thread coordination
  int xidx = blkX+tx;

  // effective range within this block that falls within
  // the valid range of the input data
  // used to rule out computation outside the boundary.
  int validXmin = (blkX < 0) ? -blkX : 0;
  int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;

  int W = tx-1;
  int E = tx+1;

  W = (W < validXmin) ? validXmin : W;
  E = (E > validXmax) ? validXmax : E;

  bool isValid = IN_RANGE(tx, validXmin, validXmax);

  if(IN_RANGE(xidx, 0, cols-1))
  {
    prev[tx] = gpuSrc[xidx];
  }

  __syncthreads();

  bool computed;
  for (int i = 0; i < iteration; i++)
  {
    computed = false;

    if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) && isValid )
    {
      computed = true;
      int left = prev[W];
      int up = prev[tx];
      int right = prev[E];
      int shortest = MIN(left, up);
      shortest = MIN(shortest, right);

      int index = cols*(t+i)+xidx;
      result[tx] = shortest + gpuWall[index];

      // ===================================================================
      // add debugging info to the debug output buffer...
      if (tx==11 && i==0)
      {
        // set bufIndex to what value/range of values you want to know.
        int bufIndex = gpuSrc[xidx];
        // dont touch the line below.
        outputBuffer[bufIndex] = 1;
      }
      // ===================================================================
    }

    __syncthreads();

    if(i==iteration-1)
    {
      // we are on the last iteration, and thus don't need to 
      // compute for the next step.
      break;
    }

    if(computed)
    {
      //Assign the computation range
      prev[tx] = result[tx];
    }
    __syncthreads();
  }

  // update the global memory
  // after the last iteration, only threads coordinated within the
  // small block perform the calculation and switch on "computed"
  if (computed)
  {
    gpuResult[xidx] = result[tx];
  }
}

int main(int argc, char** argv)
{
  // Program variables.
  int   rows, cols;
  int*  data;
  int** wall;
  int*  result;
  int   pyramid_height;

  if (argc == 4)
  {
    cols = atoi(argv[1]);
    rows = atoi(argv[2]);
    pyramid_height = atoi(argv[3]);
  }
  else
  {
    printf("Usage: %s <column length> <row length> <pyramid_height>\n", argv[0]);
    exit(0);
  }

  data = new int[rows * cols];
  wall = new int*[rows];
  for (int n = 0; n < rows; n++)
  {
    // wall[n] is set to be the nth row of the data array.
    wall[n] = data + cols * n;
  }
  result = new int[cols];

  int seed = M_SEED;
  srand(seed);

  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      wall[i][j] = rand() % 10;
    }
  }
#ifdef BENCH_PRINT
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      printf("%d ", wall[i][j]);
    }
    printf("\n");
  }
#endif

  // Pyramid parameters.
  const int borderCols = (pyramid_height) * HALO;

  /* printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",
     pyramid_height, cols, borderCols, NUMBER_THREADS, blockCols, smallBlockCol); */

  int size = rows * cols; // the size (global work size) is a multiple of lws 

  // running the opencl application shows lws=4000 (cpu) and lws=250 (gpu)
  int lws = 250;
  int* outputBuffer = (int*)calloc(16384, sizeof(int));
  int theHalo = HALO;

  double offload_start = get_time();

  int* d_gpuWall;
  cudaMalloc((void**)&d_gpuWall, sizeof(int)*(size-cols));
  cudaMemcpy(d_gpuWall, data+cols, sizeof(int)*(size-cols), cudaMemcpyHostToDevice);

  int* d_gpuSrc;
  cudaMalloc((void**)&d_gpuSrc, sizeof(int)*cols);
  cudaMemcpy(d_gpuSrc, data, sizeof(int)*cols, cudaMemcpyHostToDevice);

  int* d_gpuResult;
  cudaMalloc((void**)&d_gpuResult, sizeof(int)*cols);

  int* d_outputBuffer;
  cudaMalloc((void**)&d_outputBuffer, sizeof(int)*16384);

  dim3 gridDim (size/lws);
  dim3 blockDim (lws);

  cudaDeviceSynchronize();
  double kstart = get_time();

  for (int t = 0; t < rows - 1; t += pyramid_height)
  {
    // Calculate this for the kernel argument...
    int iteration = MIN(pyramid_height, rows-t-1);

    pathfinder<<<gridDim, blockDim>>>(
        d_gpuWall, d_gpuSrc, d_gpuResult, d_outputBuffer, 
        iteration, theHalo, borderCols, cols, t);

    int* temp = d_gpuResult;
    d_gpuResult = d_gpuSrc;
    d_gpuSrc = temp;
  }

  cudaDeviceSynchronize();
  double kend = get_time();
  printf("Total kernel execution time: %lf (s)\n", kend - kstart);

  cudaMemcpy(result, d_gpuSrc, sizeof(int)*cols, cudaMemcpyDeviceToHost);
  cudaMemcpy(outputBuffer, d_outputBuffer, sizeof(int)*16348, cudaMemcpyDeviceToHost);

  cudaFree(d_gpuResult);
  cudaFree(d_gpuSrc);
  cudaFree(d_gpuWall);
  cudaFree(d_outputBuffer);

  double offload_end = get_time();
  printf("Device offloading time = %lf (s)\n", offload_end - offload_start);

  // add a null terminator at the end of the string.
  outputBuffer[16383] = '\0';

#ifdef BENCH_PRINT
  for (int i = 0; i < cols; i++)
    printf("%d ", data[i]);
  printf("\n");
  for (int i = 0; i < cols; i++)
    printf("%d ", result[i]);
  printf("\n");
#endif

  // Memory cleanup here.
  delete[] data;
  delete[] wall;
  delete[] result;
  free(outputBuffer);

  return EXIT_SUCCESS;
}
