/*
   Illinois Open Source License

   University of Illinois/NCSA
   Open Source License

   Copyright © 2009,    University of Illinois.  All rights reserved.

   Developed by: 
   Innovative Systems Lab
   National Center for Supercomputing Applications
http://www.ncsa.uiuc.edu/AboutUs/Directorates/ISL.html

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal with the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

 * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimers.

 * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimers in the documentation and/or other materials provided with the distribution.

 * Neither the names of Innovative Systems Lab and National Center for Supercomputing Applications, nor the names of its contributors may be used to endorse or promote products derived from this Software without specific prior written permission.

 THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE SOFTWARE.
 */

#ifndef _HISTOGRAM_KERNEL_H_
#define _HISTOGRAM_KERNEL_H_

// 64 bin histogram kernel, based on nVidia whitepaper.

// NUMBINS: Number of bins, should be <= 64 on G8x hardware due to memory.
// NUMTHREADS: Number of threads per block, should be 128-256 and a multiple of 32.
// MAXNUMBLOCKS: Max number of blocks (for memory limitations).
// MAXBLOCKSEND: Merge will reduce the number of blocks to this.
// DATAPERBLOCK: Number of 32 bit words processed by each block.
//         (Note: 63 is to prevent overrun in the 1 byte counters.)
// MEMPERBLOCK: Shared memory used to store per-thread sub-histograms.
// HISTOSIZE: Size of a histogram.
#define   NUMTHREADS    128
#define   MAXNUMBLOCKS  16384
#define   MAXBLOCKSEND  32
#define   DATAPERBLOCK  (NUMTHREADS * 63)
#define   MEMPERBLOCK   (NUMTHREADS * NUMBINS)
#define   HISTOSIZE     (NUMBINS * sizeof(unsigned int))

// Device-side and Host-side memory for output.
unsigned int *d_odata;
unsigned int *h_odata;

// Helper function, integer division with rounding up.
int iDivUp(int a, int b);
// Computes per-block sub-histograms and stores them in g_odata
__global__ void histoKernel(unsigned int*__restrict__ g_odata, unsigned int*__restrict__ g_idata, int size);
// Compiles per-block sub-histograms into MAXBLOCKSEND sub-histograms.
__global__ void mergeKernel(unsigned int* g_iodata, int numBlocks);


// Init and Close are used so that the histogram function can be called many times
// without having to reallocate memory each time.
// Allocate memory
void histoInit(void) {
  cudaMallocHost((void**)&h_odata, HISTOSIZE * MAXBLOCKSEND);
  cudaMalloc((void **)&d_odata, MAXNUMBLOCKS * HISTOSIZE);
}

// Free memory
void histoClose(void) {
  cudaFree(d_odata);
  cudaFreeHost(h_odata);
}

// Host-side function; returns a NUMBINS bin histogram in h_result based off of d_idata
// IMPORTANT: d_idata is assumed to pack four bin assignments in a 32 bit integer, treating
// the 6 upper bits (0xFC) of each byte as a bin assignment. num is the number of bin assignments,
// not the number of integers in d_idata.
void GPUHistogram(unsigned int* h_result, unsigned int* d_idata, int num, cudaStream_t &stream) {
  // Determine necessary number of blocks, return if too many are needed, etc.
  const int numBlocks = iDivUp(num >> 2, DATAPERBLOCK);
  if(numBlocks > MAXNUMBLOCKS) {
    printf("Data set too large for GPU Histogram.");
    return;
  }
  // Create per-block sub-histograms in device global memory.
  dim3 grid(numBlocks, 1, 1);
  dim3 threads(NUMTHREADS, 1, 1);
  histoKernel<<< grid, threads, 0, stream >>>(d_odata, d_idata, num >> 2);

  // If there are fewer blocks than MAXBLOCKSEND+1, we can move on to compiling the sub-histograms.
  const int endNumBlocks = min(MAXBLOCKSEND, numBlocks);
  if(MAXBLOCKSEND < numBlocks){
    // Otherwise we used the merge kernel to reduce the number of sub-histograms to MAXBLOCKSEND.
    grid = make_uint3(MAXBLOCKSEND, 1, 1);
    threads = make_uint3(NUMBINS, 1, 1);
    mergeKernel<<< grid, threads, 0, stream >>>(d_odata, numBlocks);
  }
  // Copy at most MAXBLOCKSEND sub-histograms to h_odata.
  cudaMemcpyAsync(h_odata, d_odata, endNumBlocks * HISTOSIZE, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  // Compile the sub-histograms in the beginning of h_odata.
  for(int i = 1; i < endNumBlocks; i++){
    for(int j = 0; j < NUMBINS; j++)
      h_odata[j] += h_odata[i * NUMBINS + j];
  }
  // Fill in data in h_result.
  for(int i = 0; i < NUMBINS; i++) {
    h_result[i] = h_odata[i];
  }
}

int iDivUp(int a, int b){
  return (a%b != 0) ? (a/b + 1) : (a/b);
}

// g_odata are actual unsigned ints; g_idata pack four bin assignments as per above
// size is actually the number of unsigned ints in g_idata now, unlike num above.
__global__ void histoKernel(unsigned int*__restrict__ g_odata, unsigned int*__restrict__ g_idata, int size) {
  // Map [31:6] bits to [31:6], [5:4] bits to [1:0], and [3:0] bits to [5:2]
  // This ensures there are no bank conflicts when accessing s_Hist below.
  // Basically, the location we write to is (threadPos + data*NUMTHREADS)/4.
  // We take this mod 16 to find the bank we write to
  // data*NUMTHREADS / 4 is congruent to 0 mod 16
  // So we write to bank (threadPos / 4) % 16, or the [5:2] bits of threadPos.
  const int threadPos = (threadIdx.x & (~63)) | ((threadIdx.x & 15) << 2) | ((threadIdx.x & 48) >> 4);

  // Stores all per-thread sub-histograms
  __shared__ unsigned char s_Hist[MEMPERBLOCK];

  // Zero them out
  for(int pos = threadIdx.x; pos < (MEMPERBLOCK >> 2); pos += blockDim.x)
    ((unsigned int *)s_Hist)[pos] = 0;

  __syncthreads();

  // Location in g_idata in which this block starts reading
  const int gStart = __mul24(blockIdx.x, DATAPERBLOCK);
  // Amount of data to be processed by this block
  const int blockData = min(size - gStart, DATAPERBLOCK);

  unsigned int dataTemp;
  for(int pos = threadIdx.x; pos < blockData; pos += blockDim.x){
    // Read in integer from global memory, increment appropriate bins in shared memory.
    dataTemp = g_idata[gStart + pos];
    s_Hist[threadPos + __mul24( (dataTemp >>  2) & 63, NUMTHREADS)]++;
    s_Hist[threadPos + __mul24( (dataTemp >> 10) & 63, NUMTHREADS)]++;
    s_Hist[threadPos + __mul24( (dataTemp >> 18) & 63, NUMTHREADS)]++;
    s_Hist[threadPos + __mul24( (dataTemp >> 26) & 63, NUMTHREADS)]++;
  }

  __syncthreads();

  // Use NUMBINS threads to create a per-block sub-histogram from the data in shared memory
  if(threadIdx.x < NUMBINS){
    unsigned int sum = 0;
    // Each thread calculates the total number of elements in bin tid.
    const int tid = threadIdx.x;
    // Starting point in the histogram
    const int hStart = __mul24(tid, NUMTHREADS);
    // Another trick to ensure no bank conflicts. See nVidia whitepaper for more details.
    const int accumStart = (threadIdx.x & 15) * 4;

    // Iterate through thread sub-histograms' tid bins to calculate the sum.
    for(int i = 0, accum = accumStart; i < NUMTHREADS; i++){
      sum += s_Hist[hStart + accum];
      if(++accum == NUMTHREADS) accum = 0;
    }
    // Write to global memory.
    g_odata[blockIdx.x * NUMBINS + tid] = sum;
  }
}

// Merges numBlocks per-block sub-histograms into gridDim.x sub-histograms.
// (gridDim.x should be MAXBLOCKSEND for obvious reasons)
__global__ void mergeKernel(unsigned int* d_iodata, int numBlocks) {
  // Total number of histogram bins.
  const int size = numBlocks * NUMBINS;
  // (Starting) Position in global memory for this thread.
  const int gPos = blockIdx.x * NUMBINS + threadIdx.x;
  // Number of threads
  const int numThreads = gridDim.x * blockDim.x;
  unsigned int sum = 0;

  // Compute bin counts for new sub-histograms
  for(int pos = gPos; pos < size; pos += numThreads)
    sum += d_iodata[pos];

  // Write to memory, overwriting the (now useless) first portion of d_iodata.
  d_iodata[gPos] = sum;
}

#endif
