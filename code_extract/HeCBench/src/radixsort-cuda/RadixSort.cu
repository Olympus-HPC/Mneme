/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include "RadixSort.h"
#include "RadixSort_kernels.cu"

//----------------------------------------------------------------------------
// Wrapper for the kernels of the four steps
//----------------------------------------------------------------------------
void radixSortBlocksKeysOnly(
    unsigned int* d_keys, 
    unsigned int* d_tempKeys, 
    const unsigned int nbits, 
    const unsigned int startbit, 
    const unsigned int numElements)
{
  unsigned int totalBlocks = numElements/4/CTA_SIZE;
  dim3 gws (totalBlocks);
  dim3 lws (CTA_SIZE);

#ifdef DEBUG
  printf("nbits: %d startbit: %d\n", nbits, startbit);
#endif

  radixSortBlocksKeysK <<< gws, lws >>> (d_keys, d_tempKeys, nbits, startbit);
}

void findRadixOffsets(
  unsigned int* d_tempKeys, 
  unsigned int* d_counters, 
  unsigned int* d_blockOffsets, 
  const unsigned int startbit, 
  const unsigned int numElements)
{
  unsigned int totalBlocks = numElements/2/CTA_SIZE;

  dim3 gws (totalBlocks);
  dim3 lws (CTA_SIZE);

  findRadixOffsetsK <<< gws, lws >>> (d_tempKeys, d_counters, d_blockOffsets, startbit, totalBlocks);
}

void reorderDataKeysOnly(
   unsigned int* d_keys, 
   unsigned int* d_tempKeys, 
   unsigned int* d_blockOffsets, 
   unsigned int* d_countersSum, 
   const unsigned int startbit, 
   const unsigned int numElements)
{
  unsigned int totalBlocks = numElements/2/CTA_SIZE;
  dim3 gws (totalBlocks);
  dim3 lws (CTA_SIZE);

  reorderDataKeysOnlyK <<< gws, lws >>> (
                        d_keys,
                        d_tempKeys,
                        d_blockOffsets,
                        d_countersSum,
                        startbit,
                        numElements,
                        totalBlocks);
}

//----------------------------------------------------------------------------
// Perform one step of the radix sort.  Sorts by "nbits" key bits per step, 
// starting at startbit.
//----------------------------------------------------------------------------
void radixSortStepKeysOnly(unsigned int* d_keys, 
                           unsigned int* d_tempKeys, 
                           unsigned int* d_counters, 
                           unsigned int* d_blockOffsets, 
                           unsigned int* d_countersSum, 
                           unsigned int* d_buffer, 
                           const unsigned int nbits, 
                           const unsigned int startbit, 
                           const unsigned int numElements, 
                           const unsigned int batchSize )
{
  // Four step algorithms from Satish, Harris & Garland 
  radixSortBlocksKeysOnly(d_keys, d_tempKeys, nbits, startbit, numElements);

#ifdef DEBUG
  unsigned int *h_tempKeys = (unsigned int*) malloc (numElements*sizeof(unsigned int));
  cudaMemcpy(h_tempKeys, d_tempKeys, numElements*sizeof(unsigned int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < numElements; i++) printf("temp key %d: %x\n", i, h_tempKeys[i]);
  free(h_tempKeys);
#endif

  findRadixOffsets(d_tempKeys, d_counters, d_blockOffsets, startbit, numElements);

#ifdef DEBUG
  unsigned int numBlocks = (numElements / (CTA_SIZE * 4));
  unsigned int *h_counters = (unsigned int*) malloc (WARP_SIZE*numBlocks*sizeof(unsigned int));
  cudaMemcpy(h_counters, d_counters, WARP_SIZE*numBlocks*sizeof(unsigned int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < WARP_SIZE*numBlocks; i++) printf("counters %d: %x\n", i, h_counters[i]);
  free(h_counters);
#endif

  scanExclusiveLarge(d_countersSum, d_counters, d_buffer,
                     batchSize, numElements/2/CTA_SIZE*16, numElements);

#ifdef DEBUG
  unsigned int *h_countersSum = (unsigned int*) malloc (WARP_SIZE*numBlocks*sizeof(unsigned int));
  cudaMemcpy(h_countersSum, d_countersSum, WARP_SIZE*numBlocks*sizeof(unsigned int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < WARP_SIZE*numBlocks; i++) printf("countersSum %d: %x\n", i, h_countersSum[i]);
  free(h_countersSum);
#endif

  reorderDataKeysOnly(d_keys, d_tempKeys, d_blockOffsets, d_countersSum, startbit, numElements);

}


//----------------------------------------------------------------------------
// Main key-only radix sort function.  Sorts in place in the keys and values 
// arrays, but uses the other device arrays as temporary storage.  All pointer 
// parameters are device pointers.  Uses cudppScan() for the prefix sum of
// radix counters.
//----------------------------------------------------------------------------
void radixSortKeys(unsigned int* d_keys, 
                   unsigned int* d_tempKeys, 
                   unsigned int* d_counters, 
                   unsigned int* d_blockOffsets, 
                   unsigned int* d_countersSum, 
                   unsigned int* d_buffer, 
                   const unsigned int numElements, 
                   const unsigned int keyBits,
                   const unsigned int batchSize)
{
  int i = 0;
  while (keyBits > i*bitStep) 
  {
    radixSortStepKeysOnly(d_keys, d_tempKeys, d_counters, d_blockOffsets, d_countersSum,
                          d_buffer, bitStep, i*bitStep, numElements, batchSize);
    i++;
  }
}
