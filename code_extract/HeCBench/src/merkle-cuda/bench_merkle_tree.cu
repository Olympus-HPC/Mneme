#include <random>
#include <chrono>
#include "bench_merkle_tree.hpp"
#include "merkle_tree.hpp"

uint64_t
benchmark_merklize_approach_1(const size_t leaf_count,
                              const size_t wg_size)
{
  const size_t leaves_size = sizeof(ulong) * leaf_count * DIGEST_SIZE;
  const size_t mds_size = sizeof(ulong4) * STATE_WIDTH * 3;
  const size_t ark_size = sizeof(ulong4) * NUM_ROUNDS * 3;

  ulong* leaves_h;
  cudaMallocHost((void**)&leaves_h, leaves_size);

  ulong* leaves_d;
  cudaMalloc((void**)&leaves_d, leaves_size);

  ulong* intermediates_d;
  cudaMalloc((void**)&intermediates_d, leaves_size);

  ulong4* mds_h;
  cudaMallocHost((void**)&mds_h, mds_size);

  ulong4* mds_d;
  cudaMalloc((void**)&mds_d, mds_size);

  ulong4* ark1_h;
  cudaMallocHost((void**)&ark1_h, ark_size);

  ulong4* ark1_d;
  cudaMalloc((void**)&ark1_d, ark_size);

  ulong4* ark2_h;
  cudaMallocHost((void**)&ark2_h, ark_size);

  ulong4* ark2_d;
  cudaMalloc((void**)&ark2_d, ark_size);

  {
    std::mt19937 gen(19937);
    std::uniform_int_distribution<uint64_t> dis(1ul, MOD);

    for (uint64_t i = 0; i < leaf_count * DIGEST_SIZE; i++) {
      *(leaves_h + i) = static_cast<ulong>(dis(gen));
    }
  }

  prepare_mds(mds_h);
  prepare_ark1(ark1_h);
  prepare_ark2(ark2_h);

  cudaMemcpy(leaves_d, leaves_h, leaves_size, cudaMemcpyHostToDevice);
  cudaMemcpy(mds_d, mds_h, mds_size, cudaMemcpyHostToDevice);
  cudaMemcpy(ark1_d, ark1_h, ark_size, cudaMemcpyHostToDevice);
  cudaMemcpy(ark2_d, ark2_h, ark_size, cudaMemcpyHostToDevice);

  // this itself does host synchronization
  auto start_time = std::chrono::high_resolution_clock::now();
  merklize_approach_1(
    leaves_d, intermediates_d, leaf_count, wg_size, mds_d, ark1_d, ark2_d);
  auto end_time = std::chrono::high_resolution_clock::now();
  uint64_t ts = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time).count();

#ifdef DEBUG
  ulong *intermediates_h = (ulong*) malloc (leaves_size);
  cudaMemcpy(intermediates_h, intermediates_d, leaves_size, cudaMemcpyDeviceToHost);
  for (int i = 0; i < leaf_count * DIGEST_SIZE; i++) {
    printf("%d %lu\n", i, intermediates_h[i]);
  }
  free(intermediates_h);
#endif

  cudaFreeHost(leaves_h);
  cudaFreeHost(mds_h);
  cudaFreeHost(ark1_h);
  cudaFreeHost(ark2_h);

  cudaFree(leaves_d);
  cudaFree(intermediates_d);
  cudaFree(mds_d);
  cudaFree(ark1_d);
  cudaFree(ark2_d);

  return ts;
}
