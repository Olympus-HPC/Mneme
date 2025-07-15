#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <cuda.h>
#include "kernels.h"

int PowTwoDivider(int n)
{
  if (n == 0) return 0;
  int divider = 1;
  while ((n & divider) == 0) divider <<= 1;
  return divider;
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <width> <height> <repeat>\n", argv[0]);
    return 1;
  }

  const int width = atoi(argv[1]);
  const int height = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const int image_pitch = width * sizeof(float);
  const int numPix = width * height;
  const int image_size = numPix * sizeof(float);

  float *image = (float*) malloc (image_size);

  // image image with random values
  srand(123);
  for (int i = 0; i < numPix; i++) {
    uint x = rand() % 256;
    uint y = rand() % 256;
    uint z = rand() % 256;
    uint w = rand() % 256;
    *(uint*)(&image[i]) = (w << 24) | (z << 16) | (y << 8) | x;
  }

  float *d_image;
  cudaMalloc((void**)&d_image, image_size);

  int blocks = std::min(PowTwoDivider(height), 64);
  dim3 dimBlockX (blocks);
  dim3 dimGridX ((height + blocks - 1) / blocks);

  blocks = std::min(PowTwoDivider(width), 64);
  dim3 dimBlockY (blocks);
  dim3 dimGridY ((width  + blocks - 1) / blocks);

  long total_time = 0;
  for (int i = 0; i < repeat; i++) {
    cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice);
    auto start = std::chrono::steady_clock::now();

    toCoef2DX<<<dimGridX, dimBlockX>>>(d_image, image_pitch, width, height);
    toCoef2DY<<<dimGridY, dimBlockY>>>(d_image, image_pitch, width, height);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;
  }
  printf("Average kernel execution time %f (s)\n", total_time * 1e-9f / repeat);

  cudaMemcpy(image, d_image, image_size, cudaMemcpyDeviceToHost);
  cudaFree(d_image);

  float sum = 0.f;
  for (int i = 0; i < numPix; i++) {
    const uchar *t = (const uchar*)(&image[i]);
    sum += (t[0] + t[1] + t[2] + t[3]) / 4;
  }
  printf("Checksum: %f\n", sum / numPix);

  free(image);
  return 0;
}
