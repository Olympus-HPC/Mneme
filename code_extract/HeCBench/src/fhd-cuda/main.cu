#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda.h>

#define CHUNK_S 4096

typedef struct {
  float x, y, z;
} kdata;

__constant__ kdata k[CHUNK_S];

__global__
void cmpfhd(const float*__restrict__ rmu, 
            const float*__restrict__ imu,
                  float*__restrict__ rfhd,
                  float*__restrict__ ifhd,
            const float*__restrict__ x, 
            const float*__restrict__ y,
            const float*__restrict__ z,
            const int samples,
            const int voxels) 
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;

  if (n < samples) {
    float xn = x[n], yn = y[n], zn = z[n];
    float rfhdn = rfhd[n], ifhdn = ifhd[n];
    for (int m = 0; m < voxels; m++) {
      float e = 2.f * (float)M_PI * (k[m].x * xn + k[m].y * yn + k[m].z * zn);
      float c = __cosf(e);
      float s = __sinf(e);
      rfhdn += rmu[m] * c - imu[m] * s;
      ifhdn += imu[m] * c + rmu[m] * s;
    }
    rfhd[n] = rfhdn, ifhd[n] = ifhdn;
  }
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <#samples> <#voxels> <verify>\n", argv[0]);
    exit(1);
  }
  const int samples = atoi(argv[1]); // in the order of 100000
  const int voxels = atoi(argv[2]);  // cube(128)/2097152
  const int verify = atoi(argv[3]);
  const int sampleSize = samples * sizeof(float);
  const int voxelSize = voxels * sizeof(float);

  float *h_rmu = (float*) malloc (voxelSize);
  float *h_imu = (float*) malloc (voxelSize);
  float *h_kx = (float*) malloc (voxelSize);
  float *h_ky = (float*) malloc (voxelSize);
  float *h_kz = (float*) malloc (voxelSize);
  kdata *h_k = (kdata*) malloc (voxels * sizeof(kdata));

  float *h_rfhd = (float*) malloc (sampleSize);
  float *h_ifhd = (float*) malloc (sampleSize);
  float *h_x = (float*) malloc (sampleSize);
  float *h_y = (float*) malloc (sampleSize);
  float *h_z = (float*) malloc (sampleSize);

  // For device results
  float *rfhd = (float*) malloc (sampleSize);
  float *ifhd = (float*) malloc (sampleSize);

  srand(2);
  for (int i = 0; i < samples; i++) {
    h_rfhd[i] = (float)i/samples;
    h_ifhd[i] = (float)i/samples;
    h_x[i] = 0.3f + (rand()%2 ? 0.1 : -0.1);
    h_y[i] = 0.2f + (rand()%2 ? 0.1 : -0.1);
    h_z[i] = 0.1f + (rand()%2 ? 0.1 : -0.1);
  }

  for (int i = 0; i < voxels; i++) {
    h_rmu[i] = (float)i/voxels;
    h_imu[i] = (float)i/voxels;
    h_k[i].x = h_kx[i] = 0.1f + (rand()%2 ? 0.1 : -0.1);
    h_k[i].y = h_ky[i] = 0.2f + (rand()%2 ? 0.1 : -0.1);
    h_k[i].z = h_kz[i] = 0.3f + (rand()%2 ? 0.1 : -0.1);
  }

  printf("Run FHd on a device\n");
  float *d_rmu, *d_imu;
  float *d_rfhd, *d_ifhd;
  float *d_x, *d_y, *d_z;

  cudaMalloc((void**)&d_rmu, voxelSize);
  cudaMemcpy(d_rmu, h_rmu, voxelSize, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_imu, voxelSize);
  cudaMemcpy(d_imu, h_imu, voxelSize, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_rfhd, sampleSize);
  cudaMemcpy(d_rfhd, h_rfhd, sampleSize, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_ifhd, sampleSize);
  cudaMemcpy(d_ifhd, h_ifhd, sampleSize, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_x, sampleSize);
  cudaMemcpy(d_x, h_x, sampleSize, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_y, sampleSize);
  cudaMemcpy(d_y, h_y, sampleSize, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_z, sampleSize);
  cudaMemcpy(d_z, h_z, sampleSize, cudaMemcpyHostToDevice);
  
  const int ntpb = 256;
  const int nblks = (samples + ntpb - 1) / ntpb;
  dim3 grid (nblks);
  dim3 block (ntpb);

  int c = CHUNK_S;
  int s = sizeof(kdata) * c;
  int nchunks = (voxels + c - 1) / c;

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < nchunks; i++) {
    if (i == nchunks - 1) {
      c = voxels - CHUNK_S * i;
      s = sizeof(kdata) * c;
    }
    cudaMemcpyToSymbol(k, &h_k[i * CHUNK_S], s);

    cmpfhd<<<grid, block>>>(d_rmu + i*CHUNK_S,
                            d_imu + i*CHUNK_S, 
                            d_rfhd, d_ifhd, 
                            d_x, d_y, d_z, 
                            samples, c);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Device execution time %f (s)\n", time * 1e-9f);

  cudaMemcpy(rfhd, d_rfhd, sampleSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(ifhd, d_ifhd, sampleSize, cudaMemcpyDeviceToHost);

  if (verify) {
    printf("Computing root mean square error between host and device results.\n");
    printf("This will take a while..\n");

    #pragma omp parallel for 
    for (int n = 0; n < samples; n++) {
      float r = h_rfhd[n];
      float i = h_ifhd[n];
      #pragma omp parallel for simd reduction(+:r,i)
      for (int m = 0; m < voxels; m++) {
        float e = 2.f * (float)M_PI * 
                  (h_kx[m] * h_x[n] + h_ky[m] * h_y[n] + h_kz[m] * h_z[n]);
        float c = cosf(e);
        float s = sinf(e);
        r += h_rmu[m] * c - h_imu[m] * s;
        i += h_imu[m] * c + h_rmu[m] * s;
      }
      h_rfhd[n] = r;
      h_ifhd[n] = i;   
    }

    float err = 0.f;
    for (int i = 0; i < samples; i++) {
      err += (h_rfhd[i] - rfhd[i]) * (h_rfhd[i] - rfhd[i]) +
             (h_ifhd[i] - ifhd[i]) * (h_ifhd[i] - ifhd[i]) ;
    }
    printf("RMSE = %f\n", sqrtf(err / (2*samples)));
  }
 
  cudaFree(d_rmu);
  cudaFree(d_imu);
  cudaFree(d_rfhd);
  cudaFree(d_ifhd);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  free(h_rmu);
  free(h_imu);
  free(h_kx);
  free(h_ky);
  free(h_kz);
  free(h_k);
  free(h_rfhd);
  free(h_ifhd);
  free(rfhd);
  free(ifhd);
  free(h_x);
  free(h_y);
  free(h_z);

  return 0;
}
