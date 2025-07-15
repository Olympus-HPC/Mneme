#include <stdio.h>
#include "merkle_tree.hpp"

__global__ void MerklizeRescuePrimeApproach1Phase0 (
  const size_t output_offset,
  const ulong* __restrict__ leaves,
        ulong* __restrict__ intermediates,
  const ulong4* __restrict__ mds,
  const ulong4* __restrict__ ark1,
  const ulong4* __restrict__ ark2)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  merge(leaves + idx * (DIGEST_SIZE >> 1),
        intermediates + (output_offset + idx) * DIGEST_SIZE,
        mds,
        ark1,
        ark2);
}

__global__ void MerklizeRescuePrimeApproach1Phase1(
  const size_t offset,
        ulong* __restrict__ intermediates,
  const ulong4* __restrict__ mds,
  const ulong4* __restrict__ ark1,
  const ulong4* __restrict__ ark2)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  merge(intermediates + (offset << 1) * DIGEST_SIZE +
          idx * (DIGEST_SIZE >> 1),
        intermediates + (offset + idx) * DIGEST_SIZE,
        mds,
        ark1,
        ark2);
}
  
void merklize_approach_1(
  const ulong* leaves,
  ulong* const intermediates,
  const size_t leaf_count,
  const size_t wg_size,
  const ulong4* mds,
  const ulong4* ark1,
  const ulong4* ark2)
{
  // ensure only working with powers of 2 -many leaves
  assert((leaf_count & (leaf_count - 1)) == 0);
  // checking that requested work group size for first
  // phase of kernel dispatch is valid
  //
  // for next rounds of kernel dispatches, work group
  // size will be adapted when required !
  assert(wg_size <= (leaf_count >> 1));

  const size_t output_offset = leaf_count >> 1;

  dim3 grid0 (output_offset / wg_size);
  dim3 block0 (wg_size);

  // this is first phase of kernel dispatch, where I compute
  // ( in parallel ) all intermediate nodes just above leaves of tree
  MerklizeRescuePrimeApproach1Phase0 <<<grid0, block0>>> (
    output_offset, leaves, intermediates, mds, ark1, ark2);

  // for computing all remaining intermediate nodes, we'll need to
  // dispatch `rounds` -many kernels, where each round is data dependent
  // on just previous one
  const size_t rounds =
    static_cast<size_t>(log2(static_cast<double>(leaf_count >> 1)));

  for (size_t r = 0; r < rounds; r++) {
    const size_t offset = leaf_count >> (r + 2);
    int block_size = offset < wg_size ? offset : wg_size;
    dim3 grid1 (offset / block_size);
    dim3 block1 (block_size);

    cudaDeviceSynchronize();

    MerklizeRescuePrimeApproach1Phase1 <<<grid1, block1>>> (
        offset, intermediates, mds, ark1, ark2);
  }

  cudaDeviceSynchronize();
}
