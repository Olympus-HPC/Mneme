#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda.h>

#define  bidx  blockIdx.x
#define  tidx  threadIdx.x

#include "modP.h"

__global__ void intt_3_64k_modcrt(
        uint32 *__restrict__ dst,
  const uint64 *__restrict__ src)
{
  __shared__ uint64 buffer[512];
  register uint64 samples[8], s8[8];
  register uint32 fmem, tmem, fbuf, tbuf;
  fmem = (bidx<<9)|((tidx&0x3E)<<3)|(tidx&0x1);
  tbuf = tidx<<3;
  fbuf = ((tidx&0x38)<<3) | (tidx&0x7);
  tmem = (bidx<<9)|((tidx&0x38)<<3) | (tidx&0x7);
#pragma unroll
  for (int i=0; i<8; i++)
    samples[i] = src[fmem|(i<<1)];
  ntt8(samples);

#pragma unroll
  for (int i=0; i<8; i++)
    buffer[tbuf|i] = _ls_modP(samples[i], ((tidx&0x1)<<2)*i*3);
  __syncthreads();

#pragma unroll
  for (int i=0; i<8; i++)
    samples[i] = buffer[fbuf|(i<<3)];

#pragma unroll
  for (int i=0; i<4; i++) {
    s8[2*i] = _add_modP(samples[2*i], samples[2*i+1]);
    s8[2*i+1] = _sub_modP(samples[2*i], samples[2*i+1]);
  }

#pragma unroll
  for (int i=0; i<8; i++) {
    dst[(((tmem|(i<<3))&0xf)<<12)|((tmem|(i<<3))>>4)] =
      (uint32)(_mul_modP(s8[i], 18446462594437939201UL, valP));
  }
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const int nttLen = 64 * 1024;
  uint64 *ntt = (uint64*) malloc (nttLen*sizeof(uint64));
  uint32 *res = (uint32*) malloc (nttLen*sizeof(uint32));

  srand(123);
  for (int i = 0; i < nttLen; i++) {
    uint64 hi = rand();
    uint64 lo = rand();
    ntt[i] = (hi << 32) | lo;
  }

  uint64 *d_ntt;
  uint32 *d_res;
  cudaMalloc(&d_ntt, nttLen*sizeof(uint64));
  cudaMalloc(&d_res, nttLen*sizeof(uint32));
  cudaMemcpy(d_ntt, ntt, nttLen*sizeof(uint64), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    intt_3_64k_modcrt<<<nttLen/512, 64>>>(d_res, d_ntt);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  cudaMemcpy(res, d_res, nttLen*sizeof(uint32), cudaMemcpyDeviceToHost);

  uint64_t checksum = 0;
  for (int i = 0; i < nttLen; i++)
    checksum += res[i];
  printf("Checksum: %lu\n", checksum);

  cudaFree(d_ntt);
  cudaFree(d_res);
  free(ntt);
  free(res);
  return 0;
}
