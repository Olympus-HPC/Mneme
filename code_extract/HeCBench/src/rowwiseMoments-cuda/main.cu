#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <type_traits>
#include <cuda.h>
#include <thrust/tuple.h>
#include "utils.h"

template <typename T>
__global__
void RowwiseMomentsKernel(
    int64_t N,
    T eps,
    const T* X,
    T* mean,
    T* rstd) 
{
  using T_ACC = T;
  using WelfordType = WelfordData<T_ACC, int64_t>;
  using WelfordOp =
    WelfordOps<T_ACC, T_ACC, int64_t, thrust::pair<T_ACC, T_ACC>>;

  const int64_t i = blockIdx.x;
  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);
  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    const int64_t index = i * N + j;
    val = welford_op.reduce(val, static_cast<T_ACC>(X[index]), index);
  }

  // There will be a warning if we declare a __shared__ WelfordType array.
  // https://github.com/pytorch/pytorch/pull/13967
  __shared__ typename std::aligned_storage<
    sizeof(WelfordType),
    alignof(WelfordType)>::type val_shared[WARP_SIZE];

  WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);
  val = BlockReduce(
      val,
      welford_op,
      /*identity_element=*/WelfordType(0, 0, 0, 0),
      val_shared_ptr);

  if (threadIdx.x == 0) {
    T_ACC m1;
    T_ACC m2;
    thrust::tie(m1, m2) = welford_op.project(val);
    rstd[i] = rsqrt(m1 + static_cast<T_ACC>(eps));
    mean[i] = m2;
  }
}

int main(int argc, char* argv[])
{
  if (argc != 7) {
    printf("Usage: %s <batch> <channel> <width> <height> <group> <repeat>\n", argv[0]);
    return 1;
  }
  const int N = atoi(argv[1]);
  const int C = atoi(argv[2]);
  const int W = atoi(argv[3]);
  const int H = atoi(argv[4]);
  const int G = atoi(argv[5]);
  const int repeat = atoi(argv[6]);

  const int64_t D = C / G;

  double eps = 1e-6;

  size_t input_size = (size_t)N * C * W * H; 
  size_t input_size_bytes = input_size * sizeof(float);

  size_t output_size = N * G;
  size_t output_size_bytes = output_size * sizeof(float);

  float* h_X = (float*) malloc (input_size_bytes);
  float* h_mean = (float*) malloc (output_size_bytes);
  float* h_rstd = (float*) malloc (output_size_bytes);

  srand(123);
  for (size_t i = 0; i < input_size; i++) {
    h_X[i] = rand() / (float)RAND_MAX;
  }

  float *d_X;
  cudaMalloc((void**)&d_X, input_size_bytes);
  cudaMemcpy(d_X, h_X, input_size_bytes, cudaMemcpyHostToDevice); 

  float* d_mean, *d_rstd;
  cudaMalloc((void**)&d_mean, output_size_bytes);
  cudaMalloc((void**)&d_rstd, output_size_bytes);

  dim3 grid (N * G);
  dim3 block (kNumThreads);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    RowwiseMomentsKernel<float><<<grid, block>>>(
        D * H * W, eps, d_X, d_mean, d_rstd);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of RowwiseMoments kernel: %f (us)\n",
         (time * 1e-3f) / repeat);

  cudaMemcpy(h_mean, d_mean, output_size_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_rstd, d_rstd, output_size_bytes, cudaMemcpyDeviceToHost);

  double avg_rstd = 0.0, avg_mean = 0.0;
  for (size_t i = 0; i < output_size; i++) {
    avg_mean += h_mean[i];
    avg_rstd += h_rstd[i];
  }
  avg_rstd /= output_size;
  avg_mean /= output_size;

  printf("Checksum: mean = %lf and rstd = %lf\n", avg_mean, avg_rstd);

  cudaFree(d_X);
  cudaFree(d_mean);
  cudaFree(d_rstd);

  free(h_X);
  free(h_mean);
  free(h_rstd);
  return 0;
}
