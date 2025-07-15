#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

__global__
void simd_intrinsics(const int n,
                     const unsigned int* input,
                           unsigned int* output)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n) return;

  unsigned int a = input[i];
  unsigned int b = a + ((i % 2) ? 1 : -1);
  unsigned int c = a ^ b; 
  unsigned int r;

  r  = __vabs2(a);
  r ^= __vabs4(a);
  
  r ^= __vabsdiffs2(a, b);
  r ^= __vabsdiffs4(a, b);
  r ^= __vabsdiffu2(a, b);
  r ^= __vabsdiffu4(a, b);
  
  r ^= __vabsss2(a);
  r ^= __vabsss4(a);
  
  r ^= __vadd2(a, b);
  r ^= __vadd4(a, b);
  
  r ^= __vaddss2(a, b);
  r ^= __vaddss4(a, b);
  r ^= __vaddus2(a, b);
  r ^= __vaddus4(a, b);
  
  r ^= __vavgs2(a, b);
  r ^= __vavgs4(a, b);
  r ^= __vavgu2(a, b);
  r ^= __vavgu4(a, b);
  
  r ^= __vcmpeq2(a, b);
  r ^= __vcmpeq4(a, b);
  
  r ^= __vcmpges2(a, b);
  r ^= __vcmpges4(a, b);
  r ^= __vcmpgeu2(a, b);
  r ^= __vcmpgeu4(a, b);
  
  r ^= __vcmpgts2(a, b);
  r ^= __vcmpgts4(a, b);
  r ^= __vcmpgtu2(a, b);
  r ^= __vcmpgtu4(a, b);
  
  r ^= __vcmples2(a, b);
  r ^= __vcmples4(a, b);
  r ^= __vcmpleu2(a, b);
  r ^= __vcmpleu4(a, b);
  
  r ^= __vcmplts2(a, b);
  r ^= __vcmplts4(a, b);
  r ^= __vcmpltu2(a, b);
  r ^= __vcmpltu4(a, b);
  
  r ^= __vcmpne2(a, b);
  r ^= __vcmpne4(a, b);
  
  r ^= __vhaddu2(a, b);
  r ^= __vhaddu4(a, b);
  
  r ^= __viaddmax_s16x2(a, b, c);
  r ^= __viaddmax_s16x2_relu(a, b, c);
  r ^= __viaddmax_s32(a, b, c);
  r ^= __viaddmax_s32_relu(a, b, c);
  r ^= __viaddmax_u16x2(a, b, c);
  r ^= __viaddmax_u32(a, b, c);
  
  r ^= __viaddmin_s16x2(a, b, c);
  r ^= __viaddmin_s16x2_relu(a, b, c);
  r ^= __viaddmin_s32(a, b, c);
  r ^= __viaddmin_s32_relu(a, b, c);
  r ^= __viaddmin_u16x2(a, b, c);
  r ^= __viaddmin_u32(a, b, c);
  
  bool pred   ;
  bool pred_hi;
  bool pred_lo;

  r ^= __vibmax_s16x2(a, b, &pred_hi, &pred_lo);
  r ^= __vibmax_s32(a, b, &pred);
  r ^= __vibmax_u16x2(a, b, &pred_hi, &pred_lo);
  r ^= __vibmax_u32(a, b, &pred);
  
  r ^= __vibmin_s16x2(a, b, &pred_hi, &pred_lo);
  r ^= __vibmin_s32(a, b, &pred);
  r ^= __vibmin_u16x2(a, b, &pred_hi, &pred_lo);
  r ^= __vibmin_u32(a, b, &pred);
  
  r ^= __vimax3_s16x2(a, b, c);
  r ^= __vimax3_s16x2_relu(a, b, c);
  r ^= __vimax3_s32(a, b, c);
  r ^= __vimax3_s32_relu(a, b, c);
  r ^= __vimax3_u16x2(a, b, c);
  r ^= __vimax3_u32(a, b, c);
  
  r ^= __vimax_s16x2_relu(a, b);
  r ^= __vimax_s32_relu(a, b);
  
  r ^= __vimin3_s16x2(a, b, c);
  r ^= __vimin3_s16x2_relu(a, b, c);
  r ^= __vimin3_s32(a, b, c);
  r ^= __vimin3_s32_relu(a, b, c);
  r ^= __vimin3_u16x2(a, b, c);
  r ^= __vimin3_u32(a, b, c);
  
  r ^= __vimin_s16x2_relu(a, b);
  r ^= __vimin_s32_relu(a, b);
  
  r ^= __vmaxs2(a, b);
  r ^= __vmaxs4(a, b);
  r ^= __vmaxu2(a, b);
  r ^= __vmaxu4(a, b);
  
  r ^= __vmins2(a, b);
  r ^= __vmins4(a, b);
  r ^= __vminu2(a, b);
  r ^= __vminu4(a, b);
  
  r ^= __vneg2(a);
  r ^= __vneg4(a);
  r ^= __vnegss2(a);
  r ^= __vnegss4(a);
  
  r ^= __vsads2(a, b);
  r ^= __vsads4(a, b);
  r ^= __vsadu2(a, b);
  r ^= __vsadu4(a, b);
  
  r ^= __vseteq2(a, b);
  r ^= __vseteq4(a, b);
  
  r ^= __vsetges2(a, b);
  r ^= __vsetges4(a, b);
  r ^= __vsetgeu2(a, b);
  r ^= __vsetgeu4(a, b);
  
  r ^= __vsetgts2(a, b);
  r ^= __vsetgts4(a, b);
  r ^= __vsetgtu2(a, b);
  r ^= __vsetgtu4(a, b);
  
  r ^= __vsetles2(a, b);
  r ^= __vsetles4(a, b);
  r ^= __vsetleu2(a, b);
  r ^= __vsetleu4(a, b);
  
  r ^= __vsetlts2(a, b);
  r ^= __vsetlts4(a, b);
  r ^= __vsetltu2(a, b);
  r ^= __vsetltu4(a, b);
  
  r ^= __vsetne2(a, b);
  r ^= __vsetne4(a, b);
  
  r ^= __vsub2(a, b);
  r ^= __vsub4(a, b);
  r ^= __vsubss2(a, b);
  r ^= __vsubss4(a, b);
  r ^= __vsubus2(a, b);
  r ^= __vsubus4(a, b);

  output[i] = r;
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  const size_t input_size_bytes = n * sizeof(unsigned int); 
  const size_t output_size_bytes = n * sizeof(unsigned int); 
  unsigned int *input = (unsigned int*) malloc (input_size_bytes);
  unsigned int *output = (unsigned int*) malloc (output_size_bytes);

  for (int i = 0; i < n; i++) {
    input[i] = 0x1234aba5 ^ ((i < n/2) ? (-i) : i);
  }

  unsigned int *d_input, *d_output;

  cudaMalloc((void**)&d_input, input_size_bytes);
  cudaMemcpy(d_input, input, input_size_bytes, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_output, output_size_bytes);
  cudaMemset(d_output, 0, output_size_bytes);

  const int grid = (n + 255) / 256;
  const int block = 256;

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    simd_intrinsics<<<grid, block>>>(n, d_input, d_output);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the SIMD intrinsics kernel: %f (us)\n",
         (time * 1e-3f) / repeat);

  cudaMemcpy(output, d_output, output_size_bytes, cudaMemcpyDeviceToHost);

  unsigned int checksum = 0;
  for (int i = 0; i < n; i++) {
    checksum = checksum ^ output[i];
  }
  printf("Checksum = %x\n", checksum);

  cudaFree(d_input);
  cudaFree(d_output);

  free(input);
  free(output);

  return 0;
}
