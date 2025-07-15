#include <assert.h>
#include <stdio.h>
#include <chrono>
#include <cuda.h>
#include "reference.h"
#include "kernels.h"

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  bool *dinfo = NULL, *hinfo = NULL;
  unsigned int *h_input, *h_result;
  unsigned int *d_input, *d_result;

  int error_count[3] = {0, 0, 0};

  const int warp_size = 32;
  h_input = (unsigned int *)malloc(VOTE_DATA_GROUP * warp_size *
                                   sizeof(unsigned int));
  h_result = (unsigned int *)malloc(VOTE_DATA_GROUP * warp_size *
                                    sizeof(unsigned int));
  genVoteTestPattern(h_input, VOTE_DATA_GROUP * warp_size);

  cudaMalloc((void**)&d_input,
             VOTE_DATA_GROUP * warp_size * sizeof(unsigned int));
  cudaMalloc((void**)&d_result,
             VOTE_DATA_GROUP * warp_size * sizeof(unsigned int));
  cudaMemcpy(d_input, h_input,
             VOTE_DATA_GROUP * warp_size * sizeof(unsigned int),
             cudaMemcpyHostToDevice);

  dim3 gridBlock(1);
  dim3 threadBlock(VOTE_DATA_GROUP * warp_size);

  // Start of Vote Any Test Kernel #1
  printf("\tRunning <<Vote.Any>> kernel1 ...\n");

  // Warmup
  VoteAnyKernel1<<<gridBlock, threadBlock>>>(d_input, d_result, repeat);
  cudaDeviceSynchronize();

  auto start = std::chrono::steady_clock::now();

  VoteAnyKernel1<<<gridBlock, threadBlock>>>(d_input, d_result, repeat);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("\tkernel execution time: %f (s)\n", time * 1e-9f);

  cudaMemcpy(h_result, d_result,
             VOTE_DATA_GROUP * warp_size * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);

  error_count[0] += checkResultsVoteAnyKernel1(
      h_result, VOTE_DATA_GROUP * warp_size, warp_size);

  // Start of Vote All Test Kernel #2
  printf("\tRunning <<Vote.All>> kernel2 ...\n");

  // Warmup
  VoteAllKernel2<<<gridBlock, threadBlock>>>(d_input, d_result, repeat);
  cudaDeviceSynchronize();

  start = std::chrono::steady_clock::now();

  VoteAllKernel2<<<gridBlock, threadBlock>>>(d_input, d_result, repeat);

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("\tkernel execution time: %f (s)\n", time * 1e-9f);

  cudaMemcpy(h_result, d_result,
             VOTE_DATA_GROUP * warp_size * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);

  error_count[1] += checkResultsVoteAllKernel2(
      h_result, VOTE_DATA_GROUP * warp_size, warp_size);

  // Second Vote Kernel Test #3 (both Any/All)
  cudaMalloc((void**)&dinfo, warp_size * 3 * 3 * sizeof(bool));

  // Warmup
  VoteAnyKernel3<<<1, warp_size * 3>>>(dinfo, warp_size, repeat);
  cudaDeviceSynchronize();

  cudaMemset(dinfo, 0, warp_size * 3 * 3 * sizeof(bool));

  printf("\tRunning <<Vote.Any>> kernel3 ...\n");

  start = std::chrono::steady_clock::now();

  VoteAnyKernel3<<<1, warp_size * 3>>>(dinfo, warp_size, repeat);

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("\tkernel execution time: %f (s)\n", time * 1e-9f);

  hinfo = (bool*) malloc (warp_size * 3 * 3 * sizeof(bool));
  cudaMemcpy(hinfo, dinfo, warp_size * 3 * 3 * sizeof(bool),
             cudaMemcpyDeviceToHost);

  error_count[2] = checkResultsVoteAnyKernel3(hinfo, warp_size * 3);

  cudaFree(d_input);
  cudaFree(d_result);
  cudaFree(dinfo);

  free(h_input);
  free(h_result);
  free(hinfo);

  return (error_count[0] == 0 && error_count[1] == 0 && error_count[2] == 0)
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
