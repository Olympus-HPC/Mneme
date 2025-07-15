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

#include "Scan.h"
#include "Scan_kernels.cu"


void scanExclusiveLocal1(
    unsigned int* d_Dst,
    unsigned int* d_Src,
    const unsigned int n,
    const unsigned int size)
{
  size_t localWorkSize = WORKGROUP_SIZE;
  size_t globalWorkSize = (n * size) / 4;
  dim3 gws (globalWorkSize/localWorkSize);
  dim3 lws (localWorkSize);

  scanExclusiveLocal1K <<< gws, lws >>> (d_Dst, d_Src, size);
}

void scanExclusiveLocal2(
    unsigned int* d_Buffer,
    unsigned int* d_Dst,
    unsigned int* d_Src,
    const unsigned int n,
    const unsigned int size)
{
  const unsigned int elements = n * size;
  size_t localWorkSize = WORKGROUP_SIZE;
  size_t globalWorkSize = iSnapUp(elements, WORKGROUP_SIZE);
  dim3 gws (globalWorkSize/localWorkSize);
  dim3 lws (localWorkSize);

  scanExclusiveLocal2K <<< gws, lws >>> (d_Buffer, d_Dst, d_Src, elements, size);
}

void uniformUpdate(
    unsigned int* d_Dst,
    unsigned int* d_Buf,
    const unsigned int n)
{
  dim3 gws (n);
  dim3 lws (WORKGROUP_SIZE);

  uniformUpdateK <<< gws, lws >>> (d_Dst, d_Buf);
}

// main exclusive scan routine
void scanExclusiveLarge(
    unsigned int* d_Dst,
    unsigned int* d_Src,
    unsigned int* d_Buf,
    const unsigned int batchSize,
    const unsigned int arrayLength,
    const unsigned int numElements)
{

#ifdef DEBUG
  unsigned int CTA_SIZE = 128;
  unsigned int WARP_SIZE = 32;
  assert(numElements == arrayLength / 16 * CTA_SIZE * 2);
  unsigned int numBlocks = (numElements / (CTA_SIZE * 4));
#endif


  scanExclusiveLocal1(
      d_Dst,
      d_Src,
      (batchSize * arrayLength) / (4 * WORKGROUP_SIZE),
      4 * WORKGROUP_SIZE
      );

#ifdef DEBUG
  unsigned int *h_countersSum = (unsigned int*) malloc (WARP_SIZE*numBlocks*sizeof(unsigned int));
  cudaMemcpy(h_countersSum, d_Dst, WARP_SIZE*numBlocks*sizeof(unsigned int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < WARP_SIZE*numBlocks; i++) printf("local1 %d: %x\n", i, h_countersSum[i]);
#endif

  scanExclusiveLocal2(
      d_Buf,
      d_Dst,
      d_Src,
      batchSize,
      arrayLength / (4 * WORKGROUP_SIZE)
      );

#ifdef DEBUG
  unsigned int *h_buffer = (unsigned int*) malloc (sizeof(unsigned int) * (arrayLength / MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE));
  cudaMemcpy(h_buffer, d_Buf, (arrayLength / MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE)*sizeof(unsigned int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < arrayLength / MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE; i++) printf("local2 %d: %x\n", i, h_buffer[i]);
  free(h_buffer);
#endif

  uniformUpdate(
      d_Dst,
      d_Buf,
      (batchSize * arrayLength) / (4 * WORKGROUP_SIZE)
         );

#ifdef DEBUG
  cudaMemcpy(h_countersSum, d_Dst, WARP_SIZE*numBlocks*sizeof(unsigned int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < WARP_SIZE*numBlocks; i++) printf("uniform %d: %x\n", i, h_countersSum[i]);
  free(h_countersSum);
#endif

}
