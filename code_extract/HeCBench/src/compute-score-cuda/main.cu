// Copyright (C) 2013-2018 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without ion, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include "options.h"
#include "scoped_ptrs.h"
using namespace aocl_utils;

#define MANUAL_VECTOR      8 
#define NUM_THREADS_PER_WG 64
#define BLOOM_1            5 
#define BLOOM_2            0x7FFFF
#define BLOOM_SIZE         14
#define docEndingTag       0xFFFFFFFF

// Params
uint block_size = 64;
uint repeat = 100;
uint total_num_docs = 256*1024;
uint total_doc_size = 0;
uint total_doc_size_no_padding = 0;

// Host Buffers
scoped_aligned_ptr<uint> h_docWordFrequencies_dimm1;
scoped_aligned_ptr<uint> h_docWordFrequencies_dimm2;
scoped_aligned_ptr<ulong> h_profileWeights;
scoped_aligned_ptr<ulong> h_docInfo;
scoped_aligned_ptr<uint> h_isWordInProfileHash;
scoped_aligned_ptr<uint> h_startingDocID;
scoped_aligned_ptr<uint> h_numItemsPerThread;
scoped_aligned_ptr<ulong> h_profileScore;
scoped_aligned_ptr<uint> h_docSizes;

static uint m_z = 1;
static uint m_w = 1;
static uint rand_desh()
{
  m_z = 36969 * (m_z & 65535) + (m_z >> 16);
  m_w = 18000 * (m_w & 65535) + (m_w >> 16);
  return (m_z << 16) + m_w;
}

double sampleNormal() 
{     
  double u = ((double) rand() / (RAND_MAX)) * 2 - 1;     
  double v = ((double) rand() / (RAND_MAX)) * 2 - 1;     
  double r = u * u + v * v;     
  if (r == 0 || r > 1) return sampleNormal();     
  double c = sqrt(-2 * log(r) / r);     
  return u * c; 
} 

#define DOC_LEN_SIGMA 100
#define AVG_DOC_LEN   350

uint get_doc_length()
{
  int len = sampleNormal() * DOC_LEN_SIGMA + AVG_DOC_LEN;
  if (len < 10) { len = 10; } // Arbitray lower bound;
  return (uint) len;
}

// High-resolution timer.
double getCurrentTimestamp() {
#ifdef _WIN32 // Windows
  // Use the high-resolution performance counter.

  static LARGE_INTEGER ticks_per_second = {};
  if(ticks_per_second.QuadPart == 0) {
    // First call - get the frequency.
    QueryPerformanceFrequency(&ticks_per_second);
  }

  LARGE_INTEGER counter;
  QueryPerformanceCounter(&counter);

  double seconds = double(counter.QuadPart) / double(ticks_per_second.QuadPart);
  return seconds;
#else         // Linux
  timespec a;
  clock_gettime(CLOCK_MONOTONIC, &a);
  return (double(a.tv_nsec) * 1.0e-9) + double(a.tv_sec);
#endif
}

void setupData()
{
  h_startingDocID.reset( total_num_docs );
  h_numItemsPerThread.reset( total_num_docs );
  h_profileScore.reset( total_num_docs );
  h_docInfo.reset( total_num_docs );

  h_docSizes.reset( total_num_docs );

  total_doc_size = 0; 	
  total_doc_size_no_padding = 0;

  for (uint i=0; i<total_num_docs; i++) {
    uint unpadded_size = get_doc_length();
    uint size = unpadded_size & (~(2*block_size-1));
    if (unpadded_size & ((2*block_size-1))) size += 2*block_size; // Multiple of block_size
    h_startingDocID[i] = total_doc_size/2;
    h_numItemsPerThread[i] = size / (2*block_size);

    ulong start_line = total_doc_size / (2*block_size);
    ulong end_line = start_line + size / (2*block_size) - 1;

    total_doc_size += size;
    total_doc_size_no_padding += unpadded_size;
    h_docSizes[i] = unpadded_size;
    h_profileScore[i] = -1;
    h_docInfo[i] = (start_line << 32) | end_line;
  }
  h_isWordInProfileHash.reset( (1L << BLOOM_SIZE) );
  h_docWordFrequencies_dimm1.reset( total_doc_size/2 );
  h_docWordFrequencies_dimm2.reset( total_doc_size/2 );

  printf("Creating Documents total_terms=%d (no_pad=%d)\n", total_doc_size, total_doc_size_no_padding);

  for (uint i=0; i<total_doc_size/2; i++) {
    h_docWordFrequencies_dimm1[i] = docEndingTag;
    h_docWordFrequencies_dimm2[i] = docEndingTag;
  }
  for (uint doci=0; doci < total_num_docs; doci++)
  {
    uint start = h_startingDocID[doci];
    uint size = h_docSizes[doci];

    for (uint i = 0; i < size/2; i++)
    {
      uint term = (rand_desh()%((1L << 24)-1));
      uint freq = (rand_desh()%254)+1;
      h_docWordFrequencies_dimm1[start + i] = (term << 8) | freq;

      term = (rand_desh()%((1L << 24)-1));
      freq = (rand_desh()%254)+1;
      h_docWordFrequencies_dimm2[start + i] = (term << 8) | freq;
    }
    if (size%2) {
      uint term = (rand_desh()%((1L << 24)-1));
      uint freq = (rand_desh()%254)+1;
      h_docWordFrequencies_dimm1[start + size/2] = (term << 8) | freq;
    }
  }

  h_profileWeights.reset( (1L << 24) );
  for (uint i=0; i<(1L << BLOOM_SIZE); i++) {
    h_isWordInProfileHash[i] = 0x0;
  }
  printf("Creating Profile\n");
  for (uint i=0; i<(1L << 24); i++) {
    h_profileWeights[i] = 0;
  }

  for (uint i=0; i<16384; i++) {
    uint entry = (rand_desh()%(1<<24));	

    h_profileWeights[entry] = 10;

    uint hash1 = entry >> BLOOM_1;  //this gives me the top 16 bits of the 24bit word id
    h_isWordInProfileHash[ hash1 >> 5 ] |= 1 << (hash1 & 0x1f);
    uint hash2 = entry & BLOOM_2;  //this gives me the bottom 16 bits of the 24bit word id
    h_isWordInProfileHash[ hash2 >> 5 ] |= 1 << (hash2 & 0x1f);
  }
}

void runOnCPU()
{
  // go through each document in turn, and compute the score
  scoped_aligned_ptr<ulong> cpu_profileScore;
  cpu_profileScore.reset( total_num_docs );
  uint total = 0;
  uint falsies = 0;
  for (uint doci=0; doci < total_num_docs; doci++)
  {

    cpu_profileScore[doci] = 0.0;
    uint start = h_startingDocID[doci];
    uint size = h_docSizes[doci];

    for (uint i = 0; i < size/2 + (size%2); i++)
    {
      uint curr_entry = h_docWordFrequencies_dimm1[start + i];
      uint frequency = curr_entry & 0x00ff;
      uint word_id = curr_entry >> 8;
      uint hash1 = word_id >> BLOOM_1;  //this gives me the top 16 bits of the 24bit word id
      bool inh1 = h_isWordInProfileHash[ hash1 >> 5 ] & ( 1 << (hash1 & 0x1f));
      uint hash2 = word_id & BLOOM_2;  //this gives me the bottom 16 bits of the 24bit word id
      bool inh2 = h_isWordInProfileHash[ hash2 >> 5 ] & ( 1 << (hash2 & 0x1f));

      if (inh1 && inh2)
      {
        total++;
        if (h_profileWeights[word_id] == 0) falsies++;
        cpu_profileScore[doci] += h_profileWeights[word_id] * (ulong)frequency;
      }
    }

    for (uint i = 0; i < size/2; i++)
    {
      uint curr_entry = h_docWordFrequencies_dimm2[start + i];
      uint frequency = curr_entry & 0x00ff;
      uint word_id = curr_entry >> 8;
      uint hash1 = word_id >> BLOOM_1;  //this gives me the top 16 bits of the 24bit word id
      bool inh1 = h_isWordInProfileHash[ hash1 >> 5 ] & ( 1 << (hash1 & 0x1f));
      uint hash2 = word_id & BLOOM_2;  //this gives me the bottom 16 bits of the 24bit word id
      bool inh2 = h_isWordInProfileHash[ hash2 >> 5 ] & ( 1 << (hash2 & 0x1f));

      if (inh1 && inh2)
      {
        total++;
        if (h_profileWeights[word_id] == 0) falsies++;
        cpu_profileScore[doci] += h_profileWeights[word_id] * (ulong)frequency;
      }
    }
  }

  printf( "total_access = %d , falsies = %d, percentage = %f hit= %g\n", \
      total, falsies, total * 1.0f / total_doc_size, (total-falsies)*1.0f/total_doc_size );
  // compare the final scores
  for (uint doci = 0; doci < total_num_docs; doci++)
  {
    if (cpu_profileScore[doci] != h_profileScore[doci]) {
      printf("FAILED\n   : doc[%d] score: CPU = %lu, Device = %lu\n", \
          doci, cpu_profileScore[doci], h_profileScore[doci]);
      return;
    }
  }
  printf( "Verification: PASS\n" );
}

__device__
ulong mulfp( ulong weight, uint freq )
{
  uint part1 = weight & 0xFFFFF;         // lower 24-bits of weight
  uint part2 = (weight >> 24) & 0xFFFF;  // next 16-bits

  uint res1 = part1 * freq;
  uint res2 = part2 * freq;

  return (ulong)res1 + (((ulong)res2) << 24);
}

__global__ void
compute ( const uint* __restrict__ docWordFrequencies_dimm1, 
          const uint* __restrict__ docWordFrequencies_dimm2,
          const ulong*__restrict__ profileWeights_dimm1,
          const ulong*__restrict__ profileWeights_dimm2,
          const uint* __restrict__ isWordInProfileHash,
                uint* __restrict__ profileScorePerGroup_highbits_dimm1,
                uint* __restrict__ profileScorePerGroup_lowbits_dimm2 )
{
  uint curr_entry[MANUAL_VECTOR];
  uint word_id[MANUAL_VECTOR];
  uint freq[MANUAL_VECTOR];
  uint hash1[MANUAL_VECTOR];
  uint hash2[MANUAL_VECTOR];
  bool is_end[MANUAL_VECTOR];
  bool make_access[MANUAL_VECTOR];

  __shared__ ulong partial[NUM_THREADS_PER_WG/MANUAL_VECTOR];

  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  ulong sum = 0;
  //#pragma unroll
  for (uint i=0; i<MANUAL_VECTOR; i++) {
    curr_entry[i] = docWordFrequencies_dimm1[gid*MANUAL_VECTOR + i]; 
    freq[i] = curr_entry[i] & 0xff;
    word_id[i] = curr_entry[i] >> 8;
    is_end[i] = curr_entry[i] == docEndingTag;
    hash1[i] = word_id[i] >> BLOOM_1;
    hash2[i] = word_id[i] & BLOOM_2;
    make_access[i] = !is_end[i] && ((isWordInProfileHash[ hash1[i] >> 5 ] >> (hash1[i] & 0x1f)) & 0x1) 
      && ((isWordInProfileHash[ hash2[i] >> 5 ] >> (hash2[i] & 0x1f)) & 0x1); 
    if (make_access[i]) {
      sum += mulfp(profileWeights_dimm1[word_id[i]],freq[i]);
    }
  }

  //#pragma unroll
  for (uint i=0; i<MANUAL_VECTOR; i++) {
    curr_entry[i] = docWordFrequencies_dimm2[gid*MANUAL_VECTOR + i]; 
    freq[i] = curr_entry[i] & 0xff;
    word_id[i] = curr_entry[i] >> 8;
    is_end[i] = curr_entry[i] == docEndingTag;
    hash1[i] = word_id[i] >> BLOOM_1;
    hash2[i] = word_id[i] & BLOOM_2;
    make_access[i] = !is_end[i] && ((isWordInProfileHash[ hash1[i] >> 5 ] >> (hash1[i] & 0x1f)) & 0x1) 
      && ((isWordInProfileHash[ hash2[i] >> 5 ] >> (hash2[i] & 0x1f)) & 0x1); 
    if (make_access[i]) {
      sum += mulfp(profileWeights_dimm2[word_id[i]],freq[i]);
    }
  }

  partial[threadIdx.x] = sum;
  __syncthreads();

  if (threadIdx.x == 0) {
    ulong4 res= *(ulong4*)&partial[0];
    ulong4 res2= *(ulong4*)&partial[4];
    ulong final_result = res.x + res.y + res.z + res.w + res2.x + res2.y + res2.z + res2.w;
    profileScorePerGroup_highbits_dimm1[blockIdx.x] = (uint) (final_result >> 32); 
    profileScorePerGroup_lowbits_dimm2[blockIdx.x] = (uint) (final_result & 0xFFFFFFFF); 
  }
}

__global__ void
reduction( const ulong* __restrict__ docInfo,
           const uint*  __restrict__ partial_highbits_dimm1,
           const uint*  __restrict__ partial_lowbits_dimm2,
                 ulong* __restrict__ result)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  ulong info = docInfo[gid];
  unsigned start = info >> 32;
  unsigned end = info & 0xFFFFFFFF;

  ulong total = 0;
  #pragma unroll 2
  for (unsigned i=start; i<=end; i++) {
    ulong upper = partial_highbits_dimm1[i];
    ulong lower = partial_lowbits_dimm2[i];
    ulong sum = (upper << 32) | lower;
    total += sum;
  }
  result[gid] = total;
}

int main(int argc, char** argv)
{
  Options options(argc, argv);
  // Optional argument to specify the problem size.
  if(options.has("n")) {
    total_num_docs = options.get<uint>("n");
  }
  printf("Total number of documents: %u\n", total_num_docs);

  if(options.has("p")) {
    repeat = options.get<uint>("p");
  }
  printf("Kernel execution count: %u\n", repeat);

  srand(2);
  printf("RAND_MAX: %d\n", RAND_MAX);
  printf("Allocating and setting up data\n");
  setupData();

  size_t local_size = (block_size / MANUAL_VECTOR); 
  size_t global_size = total_doc_size / 2 / MANUAL_VECTOR / local_size;
  size_t local_size_reduction = block_size;
  size_t global_size_reduction = total_num_docs / block_size;

  uint* d_docWordFrequencies_dimm1;
  uint* d_docWordFrequencies_dimm2;
  uint* d_partialSums_dimm1;
  uint* d_partialSums_dimm2;
  ulong* d_profileWeights_dimm1;
  ulong* d_profileWeights_dimm2;
  uint* d_isWordInProfileHash;
  ulong* d_docInfo;
  ulong* d_profileScore;

  cudaMalloc((void**)&d_docWordFrequencies_dimm1, sizeof(uint) * total_doc_size/2);
  cudaMalloc((void**)&d_docWordFrequencies_dimm2, sizeof(uint) * total_doc_size/2);
  cudaMalloc((void**)&d_partialSums_dimm1, sizeof(uint) * total_doc_size/(2*block_size));
  cudaMalloc((void**)&d_partialSums_dimm2, sizeof(uint) * total_doc_size/(2*block_size));
  cudaMalloc((void**)&d_profileWeights_dimm1, sizeof(ulong) * (1L << 24));
  cudaMalloc((void**)&d_profileWeights_dimm2, sizeof(ulong) * (1L << 24));
  cudaMalloc((void**)&d_isWordInProfileHash, sizeof(uint) * (1L << BLOOM_SIZE));
  cudaMalloc((void**)&d_docInfo, sizeof(ulong) * total_num_docs);
  cudaMalloc((void**)&d_profileScore, sizeof(ulong) * total_num_docs);

  cudaMemcpy(d_docWordFrequencies_dimm1, h_docWordFrequencies_dimm1, sizeof(uint) * total_doc_size/2, cudaMemcpyHostToDevice);
  cudaMemcpy(d_docWordFrequencies_dimm2, h_docWordFrequencies_dimm2, sizeof(uint) * total_doc_size/2, cudaMemcpyHostToDevice);
  cudaMemcpy(d_profileWeights_dimm1, h_profileWeights, sizeof(ulong) * (1L << 24), cudaMemcpyHostToDevice);
  cudaMemcpy(d_profileWeights_dimm2, h_profileWeights, sizeof(ulong) * (1L << 24), cudaMemcpyHostToDevice);
  cudaMemcpy(d_isWordInProfileHash, h_isWordInProfileHash, sizeof(uint) * (1L << BLOOM_SIZE), cudaMemcpyHostToDevice);
  cudaMemcpy(d_docInfo, h_docInfo, sizeof(ulong) * total_num_docs, cudaMemcpyHostToDevice);

  const double start_time = getCurrentTimestamp();

  for (uint i=0; i<repeat; i++) {
    compute<<<global_size, local_size>>>(
        d_docWordFrequencies_dimm1,
        d_docWordFrequencies_dimm2, 
        d_profileWeights_dimm1,
        d_profileWeights_dimm2,
        d_isWordInProfileHash,
        d_partialSums_dimm1,
        d_partialSums_dimm2);

    reduction<<<global_size_reduction, local_size_reduction>>>(
        d_docInfo, 
        d_partialSums_dimm1,
        d_partialSums_dimm2,
        d_profileScore);
  }
  cudaDeviceSynchronize();

  const double end_time = getCurrentTimestamp();
  double kernelExecutionTime = (end_time - start_time)/repeat;
  printf("======================================================\n");
  printf("Kernel Time = %f ms (averaged over %d times)\n", kernelExecutionTime * 1000.0f, repeat );
  printf("Throughput = %f\n", total_doc_size_no_padding / kernelExecutionTime / 1.0e+6f );

  cudaMemcpy(h_profileScore, d_profileScore, sizeof(ulong) * total_num_docs, cudaMemcpyDeviceToHost);
  cudaFree(d_docWordFrequencies_dimm1);
  cudaFree(d_docWordFrequencies_dimm2);
  cudaFree(d_partialSums_dimm1);
  cudaFree(d_partialSums_dimm2);
  cudaFree(d_profileWeights_dimm1);
  cudaFree(d_profileWeights_dimm2);
  cudaFree(d_isWordInProfileHash);
  cudaFree(d_docInfo);
  cudaFree(d_profileScore);
  printf("Done\n");

  runOnCPU();
}


