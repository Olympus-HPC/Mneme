/*
  Reference
  Chapter 7 in Programming massively parallel processors,
  A hands-on approach (D. Kirk and W. Hwu)
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda.h>

#define MAX_MASK_WIDTH 10
#define BLOCK_SIZE 256
#define TILE_SIZE BLOCK_SIZE

template<typename T>
__constant__ T mask [MAX_MASK_WIDTH];

template<typename T>
__global__
void conv1d(const T * __restrict__ in,
                  T * __restrict__ out,
            const int input_width,
            const int mask_width)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  T s = 0;
  int start = i - mask_width / 2;
  for (int j = 0; j < mask_width; j++) {
    if (start + j >= 0 && start + j < input_width) {
      s += in[start + j] * mask<T>[j];
    }
  }
  out[i] = s;
}

template<typename T>
__global__
void conv1d_tiled(const T *__restrict__ in,
                        T *__restrict__ out,
                  const int input_width,
                  const int mask_width)
{
  __shared__ T tile[TILE_SIZE + MAX_MASK_WIDTH - 1];
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  int n = mask_width / 2;  // last n cells of the previous tile

  // load left cells 
  int halo_left = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
  if (threadIdx.x >= blockDim.x - n)
     tile[threadIdx.x - (blockDim.x - n)] = halo_left < 0 ? 0 : in[halo_left];

  // load center cells
  tile[n + threadIdx.x] = in[blockIdx.x * blockDim.x + threadIdx.x];

  // load right cells
  int halo_right = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
  if (threadIdx.x < n)
     tile[threadIdx.x + blockDim.x + n] = halo_right >= input_width ? 0 : in[halo_right];

  __syncthreads();

  T s = 0;
  for (int j = 0; j < mask_width; j++)
    s += tile[threadIdx.x + j] * mask<T>[j];

  out[i] = s;
}

template<typename T>
__global__
void conv1d_tiled_caching(const T *__restrict__ in,
                                T *__restrict__ out,
                          const int input_width,
                          const int mask_width)
{
  __shared__ T tile[TILE_SIZE];

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  tile[threadIdx.x] = in[i];
  __syncthreads();

  int this_tile_start = blockIdx.x * blockDim.x;
  int next_tile_start = (blockIdx.x + 1) * blockDim.x;
  int start = i - (mask_width / 2);
  T s = 0;
  for (int j = 0; j < mask_width; j++) {
    int in_index = start + j;
    if (in_index >= 0 && in_index < input_width) {
      if (in_index >= this_tile_start && in_index < next_tile_start) {
        // in_index = (start + j) = (i - mask_width/2 +j) >= 0,
        // then map in_index to tile_index
        s += tile[threadIdx.x + j - (mask_width / 2)] * mask<T>[j];
      } else {
        s += in[in_index] * mask<T>[j];
      }
    }
  }
  out[i] = s;
}

template <typename T>
void reference(const T *h_in,
               const T *d_out,
               const T *mask,
               const int input_width,
               const int mask_width)
{
  bool ok = true;
  for (int i = 0; i < input_width; i++) {
    T s = 0;
    int start = i - mask_width / 2;
    for (int j = 0; j < mask_width; j++) {
      if (start + j >= 0 && start + j < input_width) {
        s += h_in[start + j] * mask[j];
      }
    }
    if (fabs(s - d_out[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
}

template <typename T>
void conv1D(const int input_width, const int mask_width, const int repeat)
{
  size_t size_bytes = input_width * sizeof(T);

  T *a, *b;
  a = (T *)malloc(size_bytes); // input
  b = (T *)malloc(size_bytes); // output

  T h_mask[MAX_MASK_WIDTH];

  for (int i = 0; i < MAX_MASK_WIDTH; i++) h_mask[i] = 1; 

  srand(123);
  for (int i = 0; i < input_width; i++) {
    a[i] = rand() % 256;
  }

  T *d_a, *d_b;
  cudaMalloc((void **)&d_a, size_bytes);
  cudaMalloc((void **)&d_b, size_bytes);

  cudaMemcpy(d_a, a, size_bytes, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(mask<T>, h_mask, mask_width * sizeof(T));

  dim3 grids (input_width / BLOCK_SIZE);
  dim3 blocks (BLOCK_SIZE);

  cudaDeviceSynchronize();

  // conv1D basic
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    conv1d <<< grids, blocks >>> (d_a, d_b, input_width, mask_width);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time of conv1d kernel: %f (us)\n",
         (time * 1e-3f) / repeat);
  cudaMemcpy(b, d_b, size_bytes, cudaMemcpyDeviceToHost);
  reference(a, b, h_mask, input_width, mask_width);

  // conv1D tiling
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    conv1d_tiled <<< grids, blocks >>> (d_a, d_b, input_width, mask_width);
  }
  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time of conv1d-tiled kernel: %f (us)\n",
         (time * 1e-3f) / repeat);
  cudaMemcpy(b, d_b, size_bytes, cudaMemcpyDeviceToHost);
  reference(a, b, h_mask, input_width, mask_width);

  // conv1D tiling and caching
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    conv1d_tiled_caching <<< grids, blocks >>> (d_a, d_b, input_width, mask_width);
  }
  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time of conv1d-tiled-caching kernel: %f (us)\n",
         (time * 1e-3f) / repeat);
  cudaMemcpy(b, d_b, size_bytes, cudaMemcpyDeviceToHost);
  reference(a, b, h_mask, input_width, mask_width);

  free(a);
  free(b);
  cudaFree(d_a);
  cudaFree(d_b);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <input_width> <repeat>\n", argv[0]);
    return 1;
  }

  int input_width = atoi(argv[1]);
  // a multiple of BLOCK_SIZE
  input_width = (input_width + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

  const int repeat = atoi(argv[2]);

  for (int mask_width = 3; mask_width < MAX_MASK_WIDTH; mask_width += 2) {
    printf("\n---------------------\n");
    printf("Mask width: %d\n", mask_width); 

    printf("1D convolution (FP64)\n");
    conv1D<double>(input_width, mask_width, repeat);

    printf("1D convolution (FP32)\n");
    conv1D<float>(input_width, mask_width, repeat);

    printf("1D convolution (INT16)\n");
    conv1D<int16_t>(input_width, mask_width, repeat);
  }

  return 0;
}
