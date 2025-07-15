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

//----------------------------------------------------------------------------
// Scans each warp in parallel ("warp-scan"), one element per thread.
// uses 2 numElements of shared memory per thread (64 = elements per warp)
//----------------------------------------------------------------------------
#define WARP_SIZE 32
__device__
unsigned int scanwarp(unsigned int val, volatile unsigned int* sData, const int maxlevel)
{
  // The following is the same as 2 * RadixSort::WARP_SIZE * warpId + threadInWarp = 
  // 64*(threadIdx.x >> 5) + (threadIdx.x & (RadixSort::WARP_SIZE - 1))
  int localId = threadIdx.x;
  int idx = 2 * localId - (localId & (WARP_SIZE - 1));
  sData[idx] = 0;
  idx += WARP_SIZE;
  sData[idx] = val;     

  if (0 <= maxlevel) { sData[idx] += sData[idx - 1]; }
  if (1 <= maxlevel) { sData[idx] += sData[idx - 2]; }
  if (2 <= maxlevel) { sData[idx] += sData[idx - 4]; }
  if (3 <= maxlevel) { sData[idx] += sData[idx - 8]; }
  if (4 <= maxlevel) { sData[idx] += sData[idx -16]; }

  return sData[idx] - val;  // convert inclusive -> exclusive
}

//----------------------------------------------------------------------------
// scan4 scans 4*RadixSort::CTA_SIZE numElements in a block (4 per thread), using 
// a warp-scan algorithm
//----------------------------------------------------------------------------
__device__
uint4 scan4(const uint4 idata, unsigned int* ptr)
{    

  unsigned int idx = threadIdx.x;

  uint4 val4 = idata;
  unsigned int sum[3];
  sum[0] = val4.x;
  sum[1] = val4.y + sum[0];
  sum[2] = val4.z + sum[1];

  unsigned int val = val4.w + sum[2];

  val = scanwarp(val, ptr, 4);
  __syncthreads();

  if ((idx & (WARP_SIZE - 1)) == WARP_SIZE - 1)
  {
    ptr[idx >> 5] = val + val4.w + sum[2];
  }
  __syncthreads();

  if (idx < WARP_SIZE)
    ptr[idx] = scanwarp(ptr[idx], ptr, 2);

  __syncthreads();

  val += ptr[idx >> 5];

  val4.x = val;
  val4.y = val + sum[0];
  val4.z = val + sum[1];
  val4.w = val + sum[2];

  return val4;
}

__device__
uint4 rank4(const uint4 preds, unsigned int* sMem, unsigned int* numtrue)
{
  int localId = threadIdx.x;
  int localSize = blockDim.x;

  uint4 address = scan4(preds, sMem);

  if (localId == localSize - 1) 
  {
    numtrue[0] = address.w + preds.w;
  }
  __syncthreads();

  uint4 rank;
  int idx = localId*4;
  rank.x = (preds.x) ? address.x : numtrue[0] + idx - address.x;
  rank.y = (preds.y) ? address.y : numtrue[0] + idx + 1 - address.y;
  rank.z = (preds.z) ? address.z : numtrue[0] + idx + 2 - address.z;
  rank.w = (preds.w) ? address.w : numtrue[0] + idx + 3 - address.w;

  return rank;
}


__global__ void radixSortBlocksKeysK(
   const unsigned int*__restrict__ keysIn,
         unsigned int*__restrict__ keysOut,
   const unsigned int nbits,
   const unsigned int startbit)
{
  int globalId = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ unsigned int numtrue[1];
  __shared__ unsigned int sMem[4*CTA_SIZE];

  uint4 key = reinterpret_cast<const uint4*>(keysIn)[globalId];

  __syncthreads();

  // radixSortBlockKeysOnly(&key, nbits, startbit, sMem, numtrue);
  int localId = threadIdx.x;
  int localSize = blockDim.x;

  for(unsigned int shift = startbit; shift < (startbit + nbits); ++shift)
  {
    uint4 lsb;
    lsb.x = !((key.x >> shift) & 0x1);
    lsb.y = !((key.y >> shift) & 0x1);
    lsb.z = !((key.z >> shift) & 0x1);
    lsb.w = !((key.w >> shift) & 0x1);

    uint4 r;

    r = rank4(lsb, sMem, numtrue);

    // This arithmetic strides the ranks across 4 CTA_SIZE regions
    sMem[(r.x & 3) * localSize + (r.x >> 2)] = key.x;
    sMem[(r.y & 3) * localSize + (r.y >> 2)] = key.y;
    sMem[(r.z & 3) * localSize + (r.z >> 2)] = key.z;
    sMem[(r.w & 3) * localSize + (r.w >> 2)] = key.w;
    __syncthreads();

    // The above allows us to read without 4-way bank conflicts:
    key.x = sMem[localId];
    key.y = sMem[localId +     localSize];
    key.z = sMem[localId + 2 * localSize];
    key.w = sMem[localId + 3 * localSize];

    __syncthreads();
  }

  //keysOut[globalId] = key;
  reinterpret_cast<uint4*>(keysOut)[globalId] = key;  
}

//----------------------------------------------------------------------------
// Given an array with blocks sorted according to a 4-bit radix group, each 
// block counts the number of keys that fall into each radix in the group, and 
// finds the starting offset of each radix in the block.  It then writes the radix 
// counts to the counters array, and the starting offsets to the blockOffsets array.
//
// Template parameters are used to generate efficient code for various special cases
// For example, we have to handle arrays that are a multiple of the block size 
// (fullBlocks) differently than arrays that are not. "loop" is used when persistent 
// CTAs are used. 
//
// By persistent CTAs we mean that we launch only as many thread blocks as can 
// be resident in the GPU and no more, rather than launching as many threads as
// we have elements. Persistent CTAs loop over blocks of elements until all work
// is complete.  This can be faster in some cases.  In our tests it is faster
// for large sorts (and the threshold is higher on compute version 1.1 and earlier
// GPUs than it is on compute version 1.2 GPUs.
//                                
//----------------------------------------------------------------------------
__global__ void findRadixOffsetsK(
    const unsigned int*__restrict__ keys,
          unsigned int*__restrict__ counters,
          unsigned int*__restrict__ blockOffsets,
    const unsigned int startbit,
    const unsigned int totalBlocks)
{
  __shared__ unsigned int  sStartPointers[16];
  __shared__ unsigned int  sRadix1[2*CTA_SIZE];

  unsigned int groupId = blockIdx.x;
  unsigned int localId = threadIdx.x;
  unsigned int groupSize = blockDim.x;
  unsigned int globalId = groupId * groupSize + localId;

  uint2 radix2 = reinterpret_cast<const uint2*>(keys)[globalId];

  sRadix1[2 * localId]     = (radix2.x >> startbit) & 0xF;
  sRadix1[2 * localId + 1] = (radix2.y >> startbit) & 0xF;

  // Finds the position where the sRadix1 entries differ and stores start 
  // index for each radix.
  if(localId < 16) 
  {
    sStartPointers[localId] = 0; 
  }
  __syncthreads();

  if((localId > 0) && (sRadix1[localId] != sRadix1[localId - 1]) ) 
  {
    sStartPointers[sRadix1[localId]] = localId;
  }
  if(sRadix1[localId + groupSize] != sRadix1[localId + groupSize - 1]) 
  {
    sStartPointers[sRadix1[localId + groupSize]] = localId + groupSize;
  }
  __syncthreads();

  if(localId < 16) 
  {
    blockOffsets[groupId*16 + localId] = sStartPointers[localId];
  }
  __syncthreads();

  // Compute the sizes of each block.
  if((localId > 0) && (sRadix1[localId] != sRadix1[localId - 1]) ) 
  {
    sStartPointers[sRadix1[localId - 1]] = 
      localId - sStartPointers[sRadix1[localId - 1]];
  }
  if(sRadix1[localId + groupSize] != sRadix1[localId + groupSize - 1] ) 
  {
    sStartPointers[sRadix1[localId + groupSize - 1]] = 
      localId + groupSize - sStartPointers[sRadix1[localId + groupSize - 1]];
  }

  if(localId == groupSize - 1) 
  {
    sStartPointers[sRadix1[2 * groupSize - 1]] = 
      2 * groupSize - sStartPointers[sRadix1[2 * groupSize - 1]];
  }
  __syncthreads();

  if(localId < 16) 
  {
    counters[localId * totalBlocks + groupId] = sStartPointers[localId];
  }
}

//----------------------------------------------------------------------------
// reorderData shuffles data in the array globally after the radix offsets 
// have been found. On compute version 1.1 and earlier GPUs, this code depends 
// on RadixSort::CTA_SIZE being 16 * number of radices (i.e. 16 * 2^nbits).
// 
// On compute version 1.1 GPUs ("manualCoalesce=true") this function ensures
// that all writes are coalesced using extra work in the kernel.  On later
// GPUs coalescing rules have been relaxed, so this extra overhead hurts 
// performance.  On these GPUs we set manualCoalesce=false and directly store
// the results.
//
// Template parameters are used to generate efficient code for various special cases
// For example, we have to handle arrays that are a multiple of the block size 
// (fullBlocks) differently than arrays that are not.  "loop" is used when persistent 
// CTAs are used. 
//
// By persistent CTAs we mean that we launch only as many thread blocks as can 
// be resident in the GPU and no more, rather than launching as many threads as
// we have elements. Persistent CTAs loop over blocks of elements until all work
// is complete.  This can be faster in some cases.  In our tests it is faster
// for large sorts (and the threshold is higher on compute version 1.1 and earlier
// GPUs than it is on compute version 1.2 GPUs.
//----------------------------------------------------------------------------
__global__ void reorderDataKeysOnlyK(
          unsigned int*__restrict__ outKeys,
    const unsigned int*__restrict__ keys,
          unsigned int*__restrict__ blockOffsets,
    const unsigned int*__restrict__ offsets,
    const unsigned int startbit,
    const unsigned int numElements,
    const unsigned int totalBlocks)
{
  __shared__ unsigned int sOffsets[16];
  __shared__ unsigned int sBlockOffsets[16];
  __shared__ uint2 sKeys2[CTA_SIZE];

  unsigned int *sKeys1 = (unsigned int*)sKeys2;

  unsigned int groupId = blockIdx.x;
  unsigned int localId = threadIdx.x;
  unsigned int groupSize = blockDim.x;
  unsigned int globalId = groupId * groupSize + localId;
 
  sKeys2[localId] = reinterpret_cast<const uint2*>(keys)[globalId];

  if(localId < 16)
  {
    sOffsets[localId]      = offsets[localId * totalBlocks + groupId];
    sBlockOffsets[localId] = blockOffsets[groupId * 16 + localId];
  }
  __syncthreads();

  unsigned int radix = (sKeys1[localId] >> startbit) & 0xF;
  unsigned int globalOffset = sOffsets[radix] + localId - sBlockOffsets[radix];

  if (globalOffset < numElements)
  {
    outKeys[globalOffset]   = sKeys1[localId];
  }

  radix = (sKeys1[localId + groupSize] >> startbit) & 0xF;
  globalOffset = sOffsets[radix] + localId + groupSize - sBlockOffsets[radix];

  if (globalOffset < numElements)
  {
    outKeys[globalOffset]   = sKeys1[localId + groupSize];
  }
}
