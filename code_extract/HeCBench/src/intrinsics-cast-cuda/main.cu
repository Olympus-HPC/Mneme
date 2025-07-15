#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

__global__
void cast1_intrinsics(const int n,
                      const double* input,
                            long long int* output)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n) return;

  int r1 = 0;
  unsigned int r2 = 0;
  long long int r3 = 0;
  unsigned long long int r4 = 0;
  
  double x = input[i];

  r1 ^= __double2hiint(x);
  r1 ^= __double2loint(x);

  r1 ^= __double2int_rd(x);
  r1 ^= __double2int_rn(x);
  r1 ^= __double2int_ru(x);
  r1 ^= __double2int_rz(x);

  r1 ^= __float2int_rd(x);
  r1 ^= __float2int_rn(x);
  r1 ^= __float2int_ru(x);
  r1 ^= __float2int_rz(x);

  r1 ^= __float_as_int(x);

  r2 ^= __double2uint_rd(x);
  r2 ^= __double2uint_rn(x);
  r2 ^= __double2uint_ru(x);
  r2 ^= __double2uint_rz(x);

  r2 ^= __float2uint_rd(x);
  r2 ^= __float2uint_rn(x);
  r2 ^= __float2uint_ru(x);
  r2 ^= __float2uint_rz(x);
  
  r2 ^= __float_as_uint(x);

  r3 ^= __double2ll_rd(x);
  r3 ^= __double2ll_rn(x);
  r3 ^= __double2ll_ru(x);
  r3 ^= __double2ll_rz(x);

  r3 ^= __float2ll_rd(x);
  r3 ^= __float2ll_rn(x);
  r3 ^= __float2ll_ru(x);
  r3 ^= __float2ll_rz(x);

  r3 ^= __double_as_longlong(x);

  r4 ^= __double2ull_rd(x);
  r4 ^= __double2ull_rn(x);
  r4 ^= __double2ull_ru(x);
  r4 ^= __double2ull_rz(x);

  r4 ^= __float2ull_rd(x);
  r4 ^= __float2ull_rn(x);
  r4 ^= __float2ull_ru(x);
  r4 ^= __float2ull_rz(x);

  output[i] = (r1 + r2) + (r3 + r4);
}


__global__
void cast2_intrinsics(const int n,
                      const long long int* input,
                            long long int* output)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n) return;

  float r1 = 0;
  double r2 = 0;
  
  long long int x = input[i];

  r1 += __hiloint2double(x >> 32, x);

  r1 += __int2float_rd(x);
  r1 += __int2float_rn(x);
  r1 += __int2float_ru(x);
  r1 += __int2float_rz(x);

  r1 += __uint2float_rd(x);
  r1 += __uint2float_rn(x);
  r1 += __uint2float_ru(x);
  r1 += __uint2float_rz(x);

  r1 += __int_as_float(x);
  r1 += __uint_as_float(x);

  r1 += __ll2float_rd(x);
  r1 += __ll2float_rn(x);
  r1 += __ll2float_ru(x);
  r1 += __ll2float_rz(x);

  r1 += __ull2float_rd(x);
  r1 += __ull2float_rn(x);
  r1 += __ull2float_ru(x);
  r1 += __ull2float_rz(x);

  r2 += __int2double_rn(x);
  r2 += __uint2double_rn(x);

  r2 += __ll2double_rd(x);
  r2 += __ll2double_rn(x);
  r2 += __ll2double_ru(x);
  r2 += __ll2double_rz(x);

  r2 += __ull2double_rd(x);
  r2 += __ull2double_rn(x);
  r2 += __ull2double_ru(x);
  r2 += __ull2double_rz(x);

  r2 += __longlong_as_double(x);

  output[i] = __double_as_longlong(r1+r2);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  const size_t input1_size_bytes = n * sizeof(double);
  const size_t output1_size_bytes = n * sizeof(long long int); 

  const size_t input2_size_bytes = n * sizeof(long long int);
  const size_t output2_size_bytes = n * sizeof(long long int); 

  double *input1 = (double*) malloc (input1_size_bytes);
  long long int *output1 = (long long int*) malloc (output1_size_bytes);

  long long int *input2 = (long long int*) malloc (input2_size_bytes);
  long long int *output2 = (long long int*) malloc (output2_size_bytes);

  for (int i = 1; i <= n; i++) {
    input1[i] = 22.44 / i;
    input2[i] = 0x403670A3D70A3D71;
  }

  double *d_input1;
  long long int *d_output1;
  long long int *d_input2;
  long long int *d_output2;

  cudaMalloc((void**)&d_input1, input1_size_bytes);
  cudaMemcpy(d_input1, input1, input1_size_bytes, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_input2, input2_size_bytes);
  cudaMemcpy(d_input2, input2, input2_size_bytes, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_output1, output1_size_bytes);
  cudaMalloc((void**)&d_output2, output2_size_bytes);

  const int grid = (n + 255) / 256;
  const int block = 256;

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    cast1_intrinsics<<<grid, block>>>(n, d_input1, d_output1);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the cast intrinsics kernel (from FP): %f (us)\n",
         (time * 1e-3f) / repeat);

  cudaMemcpy(output1, d_output1, output1_size_bytes, cudaMemcpyDeviceToHost);

  long long int checksum1 = 0;
  for (int i = 0; i < n; i++) {
    checksum1 = checksum1 ^ output1[i];
  }
  printf("Checksum = %llx\n", checksum1);

  cudaDeviceSynchronize();
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    cast2_intrinsics<<<grid, block>>>(n, d_input2, d_output2);
  }

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the cast intrinsics kernel (to FP): %f (us)\n",
         (time * 1e-3f) / repeat);

  cudaMemcpy(output2, d_output2, output2_size_bytes, cudaMemcpyDeviceToHost);

  long long int checksum2 = 0;
  for (int i = 0; i < n; i++) {
    checksum2 ^= output2[i];
  }
  printf("Checksum = %llx\n", checksum2);

  cudaFree(d_input1);
  cudaFree(d_output1);
  cudaFree(d_input2);
  cudaFree(d_output2);

  free(input1);
  free(output1);
  free(input2);
  free(output2);

  return 0;
}
