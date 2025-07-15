#include "util.h"

const char inputName256[] = "data/input_14_1_256.bin";
const char weight_winograd_Name256[] = "data/weight_winograd_256_256.bin";
const char bnBias_winograd_Name256[] = "data/bnBias_winograd_256.bin";
const char bnScale_winograd_Name256[] = "data/bnScale_winograd_256.bin";

#define d(input, i, j, Inz) ( input[Inz + i*768 + (j<<7)] )

__global__ void kernel_256_winograd_BtdB(
  const float *__restrict__ pInputs,
        float *__restrict__ pOutputs)
{
  int Inx = blockIdx.x<<2, Iny0 = blockIdx.y<<2, Part = blockIdx.z,
      Iny1 = threadIdx.y, Inz = threadIdx.x;
  int Iny = Iny0+Iny1, stride_r = 4096, stride_c = 256; // 4096 = 16*256
  int c_glb_start = Inx*stride_r + Iny*stride_c + Inz + (Part<<7), c_input = Iny1*128 + Inz;

  extern __shared__ float input[];

  int stride_768[6] = {0, 768, 1536, 2304, 3072, 3840}; // 768 = 6*128
  for (int i = 0; i < 6; i++) {
    input[c_input + stride_768[i]] = pInputs[c_glb_start + i*stride_r];
  }
  __syncthreads();

  float BTd[6];
  switch(Iny1) {
    case 0:
      for (int j = 0; j < 6; j++) {
        BTd[j] = d(input, 0, j, Inz)*4 - d(input, 2, j, Inz)*5 + d(input, 4, j, Inz);
      }
      break;
    case 1:
      for (int j = 0; j < 6; j++) {
        BTd[j] = -d(input, 1, j, Inz)*4 - d(input, 2, j, Inz)*4 + d(input, 3, j, Inz) + d(input, 4, j, Inz);
      }
      break;
    case 2:
      for (int j = 0; j < 6; j++) {
        BTd[j] = d(input, 1, j, Inz)*4 - d(input, 2, j, Inz)*4 - d(input, 3, j, Inz) + d(input, 4, j, Inz);
      }
      break;
    case 3:
      for (int j = 0; j < 6; j++) {
        BTd[j] = -d(input, 1, j, Inz)*2 - d(input, 2, j, Inz) + d(input, 3, j, Inz)*2 + d(input, 4, j, Inz);
      }
      break;
    case 4:
      for (int j = 0; j < 6; j++) {
        BTd[j] = d(input, 1, j, Inz)*2 - d(input, 2, j, Inz) - d(input, 3, j, Inz)*2 + d(input, 4, j, Inz);
      }
      break;
    case 5:
      for (int j = 0; j < 6; j++) {
        BTd[j] = d(input, 1, j, Inz)*4 - d(input, 3, j, Inz)*5 + d(input, 5, j, Inz);
      }
      break;
  }
  __syncthreads();

  int tmp_offset = Iny1*768+Inz;
  for (int i = 0; i < 6; i++) {
    input[tmp_offset + i*128] = BTd[i];
  }
  __syncthreads();

  float BTdB[6];
  switch(Iny1) {
    case 0:
      for (int i = 0; i < 6; i++) {
        BTdB[i] = 4*d(input, i, 0, Inz) - 5*d(input, i, 2, Inz) + d(input, i, 4, Inz);
      }
      break;
    case 1:
      for (int i = 0; i < 6; i++) {
        BTdB[i] = -4*d(input, i, 1, Inz) - 4*d(input, i, 2, Inz) + d(input, i, 3, Inz) + d(input, i, 4, Inz);
      }
      break;
    case 2:
      for (int i = 0; i < 6; i++) {
        BTdB[i] = 4*d(input, i, 1, Inz) - 4*d(input, i, 2, Inz) - d(input, i, 3, Inz) + d(input, i, 4, Inz);
      }
      break;
    case 3:
      for (int i = 0; i < 6; i++) {
        BTdB[i] = -2*d(input, i, 1, Inz) - d(input, i, 2, Inz) + 2*d(input, i, 3, Inz) + d(input, i, 4, Inz);
      }
      break;
    case 4:
      for (int i = 0; i < 6; i++) {
        BTdB[i] = 2*d(input, i, 1, Inz) - d(input, i, 2, Inz) - 2*d(input, i, 3, Inz) + d(input, i, 4, Inz);
      }
      break;
    case 5:
      for (int i = 0; i < 6; i++) {
        BTdB[i] = 4*d(input, i, 1, Inz) - 5*d(input, i, 3, Inz) + d(input, i, 5, Inz);
      }
      break;
  }
  __syncthreads();

  for (int i = 0; i < 6; i++) {
    pOutputs[(Iny1 + i*6)*4096 + (blockIdx.x*4+blockIdx.y)*256 + Inz + (Part<<7)] = BTdB[i];
  }
}

__global__ void kernel_256_winograd_AtIA(
  const float *__restrict__ pInputs,
  const float *__restrict__ pBiases,
  const float *__restrict__ pScales,
        float *__restrict__ pOutputs)
{
  int Tilex = blockIdx.x, Inx = threadIdx.x;
  int Tiley = blockIdx.y, Iny = threadIdx.y;
  int kz = blockIdx.z; 
  int c_input = Inx*6 + Iny;

  __shared__ float bias, scale;
  extern __shared__ float input[];

  input[c_input] = pInputs[c_input*16*256 + (Tilex*4+Tiley)*256 + kz];
  bias = pBiases[kz];
  scale = pScales[kz];
  __syncthreads();

  float tmp = 0;
  switch(Inx) {
    case 0:
      tmp = input[Iny] + input[6+Iny] + input[12+Iny] + input[18+Iny] + input[24+Iny];
      break;
    case 1:
      tmp = input[6+Iny] - input[12+Iny] + 2*input[18+Iny] - 2*input[24+Iny];
      break;
    case 2:
      tmp = input[6+Iny] + input[12+Iny] + 4*input[18+Iny] + 4*input[24+Iny];
      break;
    case 3:
      tmp = input[6+Iny] - input[12+Iny] + 8*input[18+Iny] - 8*input[24+Iny] + input[30+Iny];
      break;
  }
  __syncthreads();

  input[c_input] = tmp;
  __syncthreads();

  if (Inx > 3 || (Tilex == 3 && Inx > 1)) return;

  int x;
  float o;
  switch(Iny) {
    case 0:
      x = Inx*6;
      o = scale*(input[x]+input[x+1]+input[x+2]+input[x+3]+input[x+4]) + bias;
      pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+1)*256 + kz] = o > 0 ? o : 0;
      break;
    case 1:
      x = Inx*6;
      o = scale*(input[x+1] - input[x+2] + 2*input[x+3] - 2*input[x+4]) + bias;
      pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+2)*256 + kz] = o > 0 ? o : 0;
      break;
    case 2:
      if (Tiley == 3) break;
      x = Inx*6;
      o = scale*(input[x+1] + input[x+2] + 4*input[x+3] + 4*input[x+4]) + bias;
      pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+3)*256 + kz] = o > 0 ? o : 0;
      break;
    case 3:
      if (Tiley == 3) break;
      x = Inx*6;
      o = scale*(input[x+1] - input[x+2] + 8*input[x+3] - 8*input[x+4] + input[x+5]) + bias;
      pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+4)*256 + kz] = o > 0 ? o : 0;
      break;
  }
}

__global__ void kernel_256_OuterProduct_256(
  const float *__restrict__ A,
  const float *__restrict__ B,
        float *__restrict__ C)
{
  int Tile = blockIdx.x, Part = blockIdx.y, 
      tX = threadIdx.x, tY = threadIdx.y;
  int c_input = tY*256 + tX,
      c_kernel = c_input,
      T_offset = (Tile<<12) + (Part<<11) + c_input, B_offset = (Tile<<16) + c_kernel;

  extern __shared__ float input[];
  float *kernel = input + 2048, *out = kernel + 8192;
  int B_stride[32] = {0, 256, 512, 768, 1024, 1280, 1536, 1792, 
                      2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840,
                      4096, 4352, 4608, 4864, 5120, 5376, 5632, 5888,
                      6144, 6400, 6656, 6912, 7168, 7424, 7680, 7936};
  out[c_input] = 0.0f;
  out[c_input+1024] = 0;

  input[c_input] = A[T_offset];
  input[c_input+1024] = A[T_offset+1024];

  for (int k = 0; k < 8; k++) {
    int B_start = B_offset + (k<<13); // 32*64
    kernel[c_kernel] = B[B_start], kernel[c_kernel+1024] = B[B_start+1024];
    kernel[c_kernel+2048] = B[B_start+2048], kernel[c_kernel+3072] = B[B_start+3072];
    kernel[c_kernel+4096] = B[B_start+4096], kernel[c_kernel+5120] = B[B_start+5120];
    kernel[c_kernel+6144] = B[B_start+6144], kernel[c_kernel+7168] = B[B_start+7168];

    __syncthreads();

    float sum = 0, sum1 = 0;
    int y_tmp = (tY<<8)+(k<<5), y_tmp1 = y_tmp+1024;
    for (int j = 0; j < 32; j++) {
      sum += input[y_tmp + j] * kernel[tX + B_stride[j]];
      sum1 += input[y_tmp1 + j] * kernel[tX + B_stride[j]];
    }
    out[c_input] += sum;
    out[c_input+1024] += sum1;
    __syncthreads();
  }

  C[T_offset] = out[c_input];
  C[T_offset+1024] = out[c_input+1024];
}

void kernel_256(double &time, double &ktime) {
  float *input_ = get_parameter(inputName256, 16*16*256);
  float *input, *output, *l_weights;

  float *kernel = get_parameter(weight_winograd_Name256, 36*256*256), *t_input, *ip;
  int nInput = 16*16*256, nOutput = 16*16*256, nWeights = 36*256*256, nBias = 256,
      nTransInput = 16*6*6*256, nInnerProd = 16*6*6*256;
  float *l_bnBias, *l_bnScale, *bnBias, *bnScale;

  float result[nOutput];
  bnBias = get_parameter(bnBias_winograd_Name256, 256);
  bnScale = get_parameter(bnScale_winograd_Name256, 256);

  auto start = std::chrono::steady_clock::now();

  cudaMalloc((void **) &input, nInput<<2);
  cudaMalloc((void **) &output, nOutput<<2);
  cudaMalloc((void **) &l_weights, nWeights<<2);
  cudaMalloc((void **) &t_input, nTransInput<<2);
  cudaMalloc((void **) &ip, nInnerProd<<2);
  cudaMalloc((void **) &l_bnBias, nBias<<2);
  cudaMalloc((void **) &l_bnScale, nBias<<2);

  cudaMemset((void *) output, 0, nOutput<<2);
  cudaMemset((void *) t_input, 0, nTransInput<<2);
  cudaMemset((void *) ip, 0, nInnerProd<<2);

  cudaMemcpy(input, input_, nInput<<2, cudaMemcpyHostToDevice);
  cudaMemcpy(l_weights, kernel, nWeights<<2, cudaMemcpyHostToDevice);
  cudaMemcpy(l_bnBias, bnBias, nBias<<2, cudaMemcpyHostToDevice);
  cudaMemcpy(l_bnScale, bnScale, nBias<<2, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  auto kstart = std::chrono::steady_clock::now();

  kernel_256_winograd_BtdB <<<dim3(4, 4, 2), dim3(128, 6), (6*6*128)<<2 >>> (input, t_input);
  kernel_256_OuterProduct_256<<<dim3(36, 2), dim3(256, 4), (8*256 + 32*256 + 8*256)<<2 >>> (t_input, l_weights, ip);
  kernel_256_winograd_AtIA <<<dim3(4, 4, 256), dim3(6, 6), ((6*6)<<2)>>> (ip, l_bnBias, l_bnScale, output);

  cudaDeviceSynchronize();
  auto kend = std::chrono::steady_clock::now();
  ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();

  cudaMemcpy(result, output, nOutput<<2, cudaMemcpyDeviceToHost);

  cudaFree(t_input);
  cudaFree(ip);
  cudaFree(input);
  cudaFree(output);
  cudaFree(l_weights);
  cudaFree(l_bnScale);
  cudaFree(l_bnBias);

  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  #ifdef DEBUG
  double s = 0;
  for (int i = 0; i < nOutput; i++) {
    s += result[i];
  }
  printf("Check sum: %lf\n", s);
  #endif

  free(kernel);
  free(bnScale);
  free(bnBias);
  free(input_);
}
