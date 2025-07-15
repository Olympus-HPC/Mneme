/*
 * Modified by Neural Magic
 * Adapted from https://github.com/Vahe1994/AQLM
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>

//#define PTX

inline int ceildiv(int a, int b) { return (a + b - 1) / b; }

const int F = 9;
const int THREAD_M = 16;

__global__ void Code1x16MatVec(
    const int4* __restrict__ A, const int4* __restrict__ B,
    int4* __restrict__ C, const int4* __restrict__ codebook, const int prob_m,
    const int prob_k,
    const int4 codebook_a_sizes,  // cumulative sizes of A spanning each
                                  // codebook, at most 3 long.
    const int codebook_stride     // as int4.
) {
  int a_gl_stride = prob_k / 8 / 8;
  int a_gl_rd = (blockDim.x / warpSize) * blockIdx.x + (threadIdx.x / warpSize);
  bool pred = a_gl_rd < prob_m;

  if (pred) {
    // advance to the correct codebook, this easy because we only multiply one
    // column of the codebook.
    auto codebook_size = &codebook_a_sizes.x;
    while (a_gl_rd >= *codebook_size) {
      codebook += codebook_stride;
      ++codebook_size;
    }
  }

  int b_gl_rd = 0;
  int c_gl_wr = a_gl_rd;
  a_gl_rd = a_gl_stride * a_gl_rd + threadIdx.x % warpSize;
  int a_gl_end = a_gl_rd + a_gl_stride - threadIdx.x % warpSize;

  // We pad shared memory to avoid bank conflicts during reads
  __shared__ int4 sh_b[32 * F]; // 32 is warpSize
  float res = 0;

  int iters = (prob_k / 8 + 8 * warpSize - 1) / (8 * warpSize);
  while (iters--) {
    __syncthreads();
    for (int i = threadIdx.x; i < warpSize * 8; i += blockDim.x) {
      if (b_gl_rd + i < prob_k / 8) sh_b[F * (i / 8) + i % 8] = B[b_gl_rd + i];
    }
    __syncthreads();
    b_gl_rd += warpSize * 8;

    int b_sh_rd = F * (threadIdx.x % warpSize);
    if (pred && a_gl_rd < a_gl_end) {
      const uint16_t* enc = reinterpret_cast<const uint16_t*>(&A[a_gl_rd]);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        uint32_t dec[4];
#ifdef PTX
        // We bypass the L1 cache to avoid massive amounts of memory streaming
        // that doesn't actually help us; this brings > 2x speedup.
        asm volatile("ld.cg.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(dec[0]), "=r"(dec[1]), "=r"(dec[2]), "=r"(dec[3])
                     : "l"((void*)&codebook[enc[i]]));
#else
        int4 t = codebook[enc[i]];
        dec[0] = t.x;
        dec[1] = t.y;
        dec[2] = t.z;
        dec[3] = t.w;
#endif
        //printf("%d %d %u %u %u %u\n", blockIdx.x, threadIdx.x, dec[0], dec[1], dec[2], dec[3]);

        half2* a = reinterpret_cast<half2*>(&dec);
        half2* b = reinterpret_cast<half2*>(&sh_b[b_sh_rd]);
        half2 res2 = {};
#pragma unroll
        for (int j = 0; j < 4; j++) res2 = __hfma2(a[j], b[j], res2);
        res += __half2float(res2.x) + __half2float(res2.y);
        b_sh_rd++;
      }
      a_gl_rd += warpSize;
    }
  }

  if (pred) {
#pragma unroll
    for (int i = warpSize/2; i > 0; i /= 2) res += __shfl_down_sync(0xffffffff, res, i);
    if (threadIdx.x % warpSize == 0) {
      reinterpret_cast<__half*>(C)[c_gl_wr] = __float2half(res);
    }
  }
}

void code1x16_matvec(const void* __restrict__ A,
                     const void* __restrict__ B, void* __restrict__ C,
                     const void* __restrict__ codebook, int prob_m,
                     int prob_k, const int4 codebook_a_sizes,
                     const int codebook_stride) {
  int sms;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int waves = 0;
  int thread_m;
  do {
    waves++;
    thread_m = ceildiv(prob_m, waves * sms);
  } while (thread_m > THREAD_M);

  int blocks = ceildiv(prob_m, thread_m);
  int threads = 32 * thread_m;

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();
  
  Code1x16MatVec<<<blocks, threads, sizeof(int4) * 32 * F, 0>>>(
      (const int4*)A, (const int4*)B, (int4*)C, (const int4*)codebook, prob_m,
      prob_k, codebook_a_sizes, codebook_stride);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("kernel execution time: %f (us)\n", (time * 1e-3f) / 1);
}

int main() {

  const int b = 4;
  
  // size of each input vector is 4096
  size_t input_size = b * 4096;

  // weight matrix  X input vectors
  size_t output_size = b * 12288 / 32;

  // weight matrix
  size_t rows = 512;  // 4096 / sizeof(int4)
  size_t cols = 12288;
  size_t code_size = cols * rows;

  // codebook
  size_t entries = 65536;
  size_t codebook_size = 3 * entries * 8;
  //int codebook_partition_sizes[] = {4096, 4096, 4096};
  int4 codebook_a_sizes = make_int4(4096, 8192, 12288, 122880);
  int codebook_stride = entries * 8 * sizeof(__half) / sizeof(int4);

  //size_t scale_size = cols;

  size_t input_size_bytes = input_size * sizeof(__half); 
  size_t output_size_bytes = output_size * sizeof(__half); 
  size_t code_size_bytes = code_size * sizeof(short); 
  size_t codebook_size_bytes = codebook_size * sizeof(__half); 
  //size_t scale_size_bytes = scale_size * sizeof(__half); 

  __half *h_input, *h_output, *h_codebook;
  //__half *h_scale;
  short *h_codes;

  h_input = (__half*) malloc (input_size_bytes);
  h_output = (__half*) malloc (output_size_bytes);
  h_codes = (short*) malloc (code_size_bytes);
  h_codebook = (__half*) malloc (codebook_size_bytes);
  //h_scale = (__half*) malloc (scale_size_bytes);

  srand(123);
  for (size_t i = 0; i < input_size; i++)  
    h_input[i] = (float)i / input_size;

  for (size_t i = 0; i < code_size; i++)  
    h_codes[i] = rand() % 65536 - 32768;

  for (size_t i = 0; i < codebook_size; i++)  
    h_codebook[i] = (float)i / codebook_size;

  //for (size_t i = 0; i < scale_size; i++)  
  //  h_scale[i] = (float)i / scale_size;

  __half *d_input, *d_output, *d_codebook;
  //__half *d_scale;
  short *d_codes;

  cudaMalloc(&d_input, input_size_bytes);
  cudaMalloc(&d_output, output_size_bytes);
  cudaMalloc(&d_codes, code_size_bytes);
  cudaMalloc(&d_codebook, codebook_size_bytes);
  //cudaMalloc(&d_scale, scale_size_bytes);

  cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_codes, h_codes, code_size_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_codebook, h_codebook, codebook_size_bytes, cudaMemcpyHostToDevice);

  int prob_m = 12288;
  int prob_k = 4096;

  for (int i = 0; i < b; ++i) {
    auto d_input_vec = d_input + 4096 * i;
    auto d_output_vec = d_output + 12288/32 * i;
    code1x16_matvec((void*)d_codes, (void*)d_input_vec,
                    (void*)d_output_vec, (void*)d_codebook,
                    prob_m, prob_k, codebook_a_sizes, codebook_stride);
  }

  cudaMemcpy(h_output, d_output, output_size_bytes, cudaMemcpyDeviceToHost);

#ifdef DEBUG
  for (int i = 0; i < b; ++i) {
    auto h_output_vec = h_output + 12288/32 * i;
    for (int j = 0; j < 12288/32; j++) {
      printf("row=%d col=%d val=%f\n", i, j, __half2float(h_output_vec[j]));
    }
  }
#endif

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_codes);
  cudaFree(d_codebook);
  //cudaFree(d_scale);
  free(h_input);
  free(h_output);
  free(h_codes);
  free(h_codebook);
  //free(h_scale);
}
