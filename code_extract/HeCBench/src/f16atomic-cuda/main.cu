#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

static void CheckError( cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
  }
}
#define CHECK_ERROR( err ) (CheckError( err, __FILE__, __LINE__ ))

#define BLOCK_SIZE 256

#define ZERO_FP16 __ushort_as_half((unsigned short)0x0000U)
#define ZERO_BF16 __ushort_as_bfloat16((unsigned short)0x0000U)
#define MIN_FP16  __ushort_as_half((unsigned short)0x0400U)
#define MIN_BF16  __ushort_as_bfloat16((unsigned short)0x0080U)

__global__
void f16AtomicOnGlobalMem(__half* result, int n, int repeat)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  __half2 *result_v = reinterpret_cast<__half2*>(result);
  __half2 val {ZERO_FP16, MIN_FP16};
  for (int i = 0; i < repeat; i++)
    atomicAdd(&result_v[tid], val);
}

__global__
void f16AtomicOnGlobalMem(__nv_bfloat16* result, int n, int repeat)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  __nv_bfloat162 *result_v = reinterpret_cast<__nv_bfloat162*>(result);
  __nv_bfloat162 val {ZERO_BF16, MIN_BF16};
  for (int i = 0; i < repeat; i++)
    atomicAdd(&result_v[tid], val);
}

template <typename T>
void atomicCost (int nelems, int repeat)
{
  size_t result_size = sizeof(T) * nelems;

  T* result = (T*) malloc (result_size);

  T *d_result;
  CHECK_ERROR( cudaMalloc((void **)&d_result, result_size) );

  dim3 block (BLOCK_SIZE);
  dim3 grid ((nelems / 2  + BLOCK_SIZE - 1) / BLOCK_SIZE);

  const int atomics_count = 256;

  //  warmup
  f16AtomicOnGlobalMem<<<grid, block>>>(d_result, nelems/2, atomics_count);

  CHECK_ERROR( cudaDeviceSynchronize() );
  auto start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    f16AtomicOnGlobalMem<<<grid, block>>>(d_result, nelems/2, atomics_count);
  }
  CHECK_ERROR( cudaDeviceSynchronize() );
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of 16-bit floating-point atomic add on global memory: %f (us)\n",
          time * 1e-3f / repeat);
  CHECK_ERROR( cudaMemcpy(result, d_result, result_size, cudaMemcpyDeviceToHost) );

  printf("Print the first two elements: 0x%04x 0x%04x\n\n", result[0], result[1]);
  free(result);
  CHECK_ERROR(cudaFree(d_result));
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <N> <repeat>\n", argv[0]);
    printf("N: total number of elements (a multiple of 2)\n");
    return 1;
  }
  const int nelems = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  assert(nelems > 0 && (nelems % 2) == 0);

  // supported since ROCm 6.3.0
  printf("\nFP16 atomic add\n");
  atomicCost<__half>(nelems, repeat);

  printf("\nBF16 atomic add\n");
  atomicCost<__nv_bfloat16>(nelems, repeat);

  return 0;
}
