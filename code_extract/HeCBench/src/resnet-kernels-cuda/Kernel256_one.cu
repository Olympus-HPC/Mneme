#include "util.h"

const char inputName256one[] = "data/input_one_14_1024.bin";
const char weightName256one[] = "data/weight_one_1024.bin";
const char bnBias_myKernel_Name256one[] = "data/bnBias_myKernel_one_1024.bin";
const char bnScale_myKernel_Name256one[] = "data/bnScale_myKernel_one_1024.bin";

__global__ void kernel_1024_one_256(
  const float *__restrict__ A,
  const float *__restrict__ B,
  const float *__restrict__ bnBias,
  const float *__restrict__ bnScale,
        float *__restrict__ C) 
{
  int tile = blockIdx.x, in_channel = threadIdx.x, line = threadIdx.y;
  int ind = line*256 + in_channel;

  extern __shared__ float shared_[];
  float *__restrict__ weights = shared_ + 1024*4,
        *__restrict__ output = weights + 256*16,
        *__restrict__ input = shared_;
  float *__restrict__ bias = output + 4*256,
        *__restrict__ scale = bias + 256;

  for (int i = 0; i < 4; i++)
    input[ind + i*1024] = A[tile*4096 + i*1024 + ind];
  bias[in_channel] = bnBias[in_channel];
  scale[in_channel] = bnScale[in_channel];
  output[ind] = 0.0f;
  __syncthreads();

  for (int k = 0; k < 1024; k += 16) {
    const float *B_start = B + k*256;
    for (int i = 0; i < 4; i++)
      weights[ind + i*1024] = B_start[i*1024 + ind];
    __syncthreads();

    const float *A_start = input + k;
    for (int p = 0; p < 16; p++) {
      output[ind] += A_start[line*1024 + p] * weights[in_channel + p*256];
    }
    __syncthreads();
  }

  float *C_start = C + tile*1024, res = scale[in_channel] * output[ind] + bias[in_channel];
  C_start[ind] = res > 0 ? res : 0;
}

__global__ void kernel_256_one_1024(
  const float *__restrict__ A,
  const float *__restrict__ B,
  const float *__restrict__ bnBias,
  const float *__restrict__ bnScale,
        float *__restrict__ C) 
{
  int tile = blockIdx.x, part = blockIdx.y, in_channel = threadIdx.x, line = threadIdx.y;
  int ind = line*256 + in_channel;

  extern __shared__ float shared_[];
  float *weights = shared_ + 256*4, *output = weights + 256*32, *input = shared_;
  float *bias = output + 4*256, *scale = bias + 256;

  input[ind] = A[tile * 1024 + ind];
  bias[in_channel] = bnBias[part*256 + in_channel];
  scale[in_channel] = bnScale[part*256+ in_channel];
  output[ind] = 0.0f;
  __syncthreads();

  for (int k = 0; k < 256; k += 32) {
    for (int i = 0; i < 8; i++)
      weights[ind + 1024*i] = B[(k + i*4 + line)*1024 + part*256 + in_channel];
    __syncthreads();

    float *A_start = input + k;
    for (int p = 0; p < 32; p++) {
      output[ind] += A_start[line*256 + p] * weights[in_channel + p*256];
    }
    __syncthreads();
  }

  float *C_start = C + tile*4096 + part*256;
  C_start[line * 1024 + in_channel] = scale[in_channel] * output[ind] + bias[in_channel];
}

void kernel_256_1_in(double &time, double &ktime) {
  float *input = get_parameter(inputName256one, 14*14*1024);
  float *weight = get_parameter(weightName256one, 256*1024);
  float *bnBias_myKernel = get_parameter(bnBias_myKernel_Name256one, 256);
  float *bnScale_myKernel = get_parameter(bnScale_myKernel_Name256one, 256);

  float *input_, *output_, *weight_, *bnBias_, *bnScale_;

  int nInput = 14*14*1024, nOutput = 14*14*256, nWeights = 256*1024;
  float result[nOutput];

  auto start = std::chrono::steady_clock::now();

  cudaMalloc((void **) &input_, nInput<<2);
  cudaMalloc((void **) &output_, nOutput<<2);
  cudaMalloc((void **) &weight_, nWeights<<2);
  cudaMalloc((void **) &bnBias_, 256<<2);
  cudaMalloc((void **) &bnScale_, 256<<2);

  cudaMemcpy(input_, input, nInput<<2, cudaMemcpyHostToDevice);
  cudaMemcpy(weight_, weight, nWeights<<2, cudaMemcpyHostToDevice);
  cudaMemcpy(bnBias_, bnBias_myKernel, 256<<2, cudaMemcpyHostToDevice);
  cudaMemcpy(bnScale_, bnScale_myKernel, 256<<2, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  auto kstart = std::chrono::steady_clock::now();

  kernel_1024_one_256 <<<dim3(49), dim3(256, 4), (4*1024 + 16*256 + 4*256 + 2*256)<<2 >>> (
    input_, weight_, bnBias_, bnScale_, output_);

  cudaDeviceSynchronize();
  auto kend = std::chrono::steady_clock::now();
  ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();

  cudaMemcpy(result, output_, nOutput<<2, cudaMemcpyDeviceToHost);

  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  cudaFree(input_);
  cudaFree(output_);
  cudaFree(weight_);
  cudaFree(bnScale_);
  cudaFree(bnBias_);

  #ifdef DEBUG
  double s = 0;
  for (int i = 0; i < nOutput; i++) {
    s += result[i];
  }
  printf("Check sum: %lf\n", s);
  #endif

  free(input);
  free(weight);
  free(bnBias_myKernel);
  free(bnScale_myKernel);
}

void kernel_256_1_out(double &time, double &ktime) {
  float *input = get_parameter(inputName256one, 14*14*256);
  float *weight = get_parameter(weightName256one, 256*1024);
  float *bnBias_myKernel = get_parameter(bnBias_myKernel_Name256one, 1024);
  float *bnScale_myKernel = get_parameter(bnScale_myKernel_Name256one, 1024);

  float *input_, *output_, *weight_, *bnBias_, *bnScale_;

  int nInput = 14*14*256, nOutput = 14*14*1024, nWeights = 256*1024;
  float result[nOutput];

  auto start = std::chrono::steady_clock::now();

  cudaMalloc((void **) &input_, nInput<<2);
  cudaMalloc((void **) &output_, nOutput<<2);
  cudaMalloc((void **) &weight_, nWeights<<2);
  cudaMalloc((void **) &bnBias_, 1024<<2);
  cudaMalloc((void **) &bnScale_, 1024<<2);

  cudaMemcpy(input_, input, nInput<<2, cudaMemcpyHostToDevice);
  cudaMemcpy(weight_, weight, nWeights<<2, cudaMemcpyHostToDevice);
  cudaMemcpy(bnBias_, bnBias_myKernel, 1024<<2, cudaMemcpyHostToDevice);
  cudaMemcpy(bnScale_, bnScale_myKernel, 1024<<2, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  auto kstart = std::chrono::steady_clock::now();

  kernel_256_one_1024 <<<dim3(49, 4), dim3(256, 4), (4*256 + 32*256 + 4*256 + 2*256)<<2 >>>(
    input_, weight_, bnBias_, bnScale_, output_);

  cudaDeviceSynchronize();
  auto kend = std::chrono::steady_clock::now();
  ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();

  cudaMemcpy(result, output_, nOutput<<2, cudaMemcpyDeviceToHost);

  cudaFree(input_);
  cudaFree(output_);
  cudaFree(weight_);
  cudaFree(bnScale_);
  cudaFree(bnBias_);

  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  #ifdef DEBUG
  double s = 0;
  for (int i = 0; i < nOutput; i++) {
    s += result[i];
  }
  printf("Check sum: %lf\n", s);
  #endif

  free(bnBias_myKernel);
  free(bnScale_myKernel);
  free(input);
  free(weight);
}
