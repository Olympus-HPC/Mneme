#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda.h>
#include "util.h"
#include "image.h"
#include "kernel.h"

int main(int argc, char* argv[]) 
{
  if (argc != 4) {
    printf("Usage: %s <width> <height> <repeat>\n", argv[0]);
    return 1;
  }

  const int width = atoi(argv[1]);
  const int height = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const int input_image_pitch = width;
  const int output_image_pitch = width * 4;
  const int numPix = width * height;
  const int input_image_size = numPix * sizeof(uchar);
  const int output_image_size = numPix * 4 * sizeof(uchar);

  uchar *input = (uchar*) malloc (input_image_size);
  uchar *output = (uchar*) malloc (output_image_size);

  // RGGB, GRBG, GBRG, BGGR
  const int bayer_pattern = RGGB;

  // input image with random values
  srand(123);
  for (int i = 0; i < numPix; i++) {
    input[i] = rand() % 256;
  }

  uchar *d_input;
  cudaMalloc((void**)&d_input, input_image_size);
  cudaMemcpy(d_input, input, input_image_size, cudaMemcpyHostToDevice);

  uchar *d_output;
  cudaMalloc((void**)&d_output, output_image_size);

  dim3 grids ((width + tile_cols - 1) / tile_cols, (height + tile_rows - 1) / tile_rows);
  dim3 blocks (tile_cols, tile_rows);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  //this version takes a tile (z=1) and each tile job does 4 line median sorts
  for (int i = 0; i < repeat; i++) {
    cudaMemset(d_output, 0, output_image_size);
    malvar_he_cutler_demosaic <<< grids, blocks >>> (
      height, width, 
      d_input, input_image_pitch,
      d_output, output_image_pitch,
      bayer_pattern );
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", time * 1e-9f / repeat);

  cudaMemcpy(output, d_output, output_image_size, cudaMemcpyDeviceToHost);

  long sum = 0;
  for (int i = 0; i < numPix; i++) sum += output[i];
  printf("Checksum: %ld\n", sum);

  free(input);
  free(output);
  cudaFree(d_input);
  cudaFree(d_output);
  return 0;
}
