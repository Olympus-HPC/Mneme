#include "util.h"

const char inputName128[] = "data/input_14_1_128.bin";
const char biasName128[] = "data/bias_128.bin";
const char weight_winograd_Name128[] = "data/weight_winograd_128_128.bin";
const char bnBias_winograd_Name128[] = "data/bnBias_winograd_128.bin";
const char bnScale_winograd_Name128[] = "data/bnScale_winograd_128.bin";

#define d(input, i, j, Inz) ( input[Inz + i*768 + (j<<7)] )

__global__ void kernel_128_winograd_BtdB(
  const float *__restrict__ pInputs,
        float *__restrict__ pOutputs)
{
  int Inx = blockIdx.x<<2, Iny0 = blockIdx.y<<2, Iny1 = threadIdx.y, Inz = threadIdx.x;
  int Iny = Iny0+Iny1, stride_r = 2048, stride_c = 128; // 2048 = 16*128
  int c_glb_start = Inx*stride_r + Iny*stride_c + Inz, c_input = Iny1*stride_c + Inz;

  extern __shared__ float input[];

  int tmp[6] = {0, 768, 1536, 2304, 3072, 3840}; // 768 = 6*128
  for (int i = 0; i < 6; i++) {
    input[c_input + tmp[i]] = pInputs[c_glb_start + i*stride_r];
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
    input[tmp_offset + i*stride_c] = BTd[i];
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
    pOutputs[(Iny1 + i*6)*2048 + (blockIdx.x*4+blockIdx.y)*128 + Inz] = BTdB[i];
  }
}

__global__ void kernel_128_winograd_AtIA(
  const float *__restrict__ pInputs,
  const float *__restrict__ pBiases,
  const float *__restrict__ pScales,
        float *__restrict__ pOutputs)
{
  int Tilex = blockIdx.x, Tiley = blockIdx.y, Iny = threadIdx.y, kz = blockIdx.z, Inx = threadIdx.x;
  int c_input = Inx*6 + Iny;

  __shared__ float bias, scale;
  extern __shared__ float input[];

  input[c_input] = pInputs[c_input*16*128 + (Tilex*4+Tiley)*128 + kz];
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
      o = scale*(input[x]+input[x+1]+input[x+2]+input[x+3]+input[x+4])+ bias;
      pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+1)*128 + kz] = o > 0 ? o : 0;
      break;
    case 1:
      x = Inx*6;
      o = scale*(input[x+1] - input[x+2] + 2*input[x+3] - 2*input[x+4]) + bias;
      pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+2)*128 + kz] = o > 0 ? o : 0;
      break;
    case 2:
      if (Tiley == 3) break;
      x = Inx*6;
      o = scale*(input[x+1] + input[x+2] + 4*input[x+3] + 4*input[x+4]) + bias;
      pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+3)*128 + kz] = o > 0 ? o : 0;
      break;
    case 3:
      if (Tiley == 3) break;
      x = Inx*6;
      o = scale*(input[x+1] - input[x+2] + 8*input[x+3] - 8*input[x+4] + input[x+5]) + bias;
      pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+4)*128 + kz] = o > 0 ? o : 0;
      break;
  }
}

__global__ void kernel_128_OuterProduct_128(
  const float *__restrict__ A,
  const float *__restrict__ B,
        float *__restrict__ C)
{
  int Tile = blockIdx.x, Part = blockIdx.y, tX = threadIdx.x, tY = threadIdx.y;
  int c_input = tY*128 + tX, c_kernel = c_input;
  int T_offset = (Tile<<11) + (Part<<10) + c_input;
  int B_offset = (Tile<<14) + c_kernel;

  extern __shared__ float input[];
  float *kernel = input + 1024, *out = kernel + 8192;
  int B_stride[32] = {0, 128, 256, 384, 512, 640, 768, 896,
                      1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920,
                      2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944,
                      3072, 3200, 3328, 3456, 3584, 3712, 3840, 3968};
  out[c_input] = 0.0f;
  
  input[c_input] = A[T_offset];
  
  for (int k = 0; k < 4; k++) {
    int B_start = B_offset + (k<<12); // 32*64
    kernel[c_kernel] = B[B_start], kernel[c_kernel+1024] = B[B_start+1024];
    kernel[c_kernel+2048] = B[B_start+2048], kernel[c_kernel+3072] = B[B_start+3072];
    __syncthreads();
  
    float sum = 0;
    int y_tmp = (tY<<7)+(k<<5);
    for (int j = 0; j < 32; j++) {
      sum += input[y_tmp + j] * kernel[tX + B_stride[j]];
    }
    out[tY*128 + tX] += sum;
    __syncthreads();
  }
  
  C[T_offset] = out[c_input];
}

void kernel_128(double &time, double &ktime) {
  float *input_ = get_parameter(inputName128, 16*16*128);
  float *bias = get_parameter(biasName128, 128);
  float *input, *output, *l_weights;

  float *t_input, *ip;
  float *kernel = get_parameter(weight_winograd_Name128, 36*128*128);
  float *l_bnBias, *l_bnScale, *bnBias, *bnScale;

  int nInput = 16*16*128, nOutput = 16*16*128, nWeights = 36*128*128, nBias = 128,
      nTransInput = 16*6*6*128, nInnerProd = 16*6*6*128;

  float result[nOutput];

  bnBias = get_parameter(bnBias_winograd_Name128, 128);
  bnScale = get_parameter(bnScale_winograd_Name128, 128);

  auto start = std::chrono::steady_clock::now();

  cudaMalloc((void **) &input, nInput<<2);
  cudaMalloc((void **) &output, nOutput<<2);
  cudaMalloc((void **) &l_weights, nWeights<<2);
  cudaMalloc((void **) &t_input, nTransInput<<2);
  cudaMalloc((void **) &ip, nInnerProd<<2);

  cudaMemset((void *) output, 0, nOutput<<2);
  cudaMemset((void *) t_input, 0, nTransInput<<2);
  cudaMemset((void *) ip, 0, nInnerProd<<2);

  cudaMemcpy(input, input_, nInput<<2, cudaMemcpyHostToDevice);
  cudaMemcpy(l_weights, kernel, nWeights<<2, cudaMemcpyHostToDevice);

  cudaMalloc((void **) &l_bnBias, nBias<<2);
  cudaMalloc((void **) &l_bnScale, nBias<<2);
  cudaMemcpy(l_bnBias, bnBias, nBias<<2, cudaMemcpyHostToDevice);
  cudaMemcpy(l_bnScale, bnScale, nBias<<2, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  auto kstart = std::chrono::steady_clock::now();

  kernel_128_winograd_BtdB <<<dim3(4, 4), dim3(128, 6), (6*6*128)<<2 >>> (input, t_input);
  kernel_128_OuterProduct_128<<<dim3(36, 2), dim3(128, 8), (8*128 + 64*128 + 8*128)<<2 >>> (t_input, l_weights, ip);
  kernel_128_winograd_AtIA <<<dim3(4, 4, 128), dim3(6, 6), ((6*6)<<2)>>> (ip, l_bnBias, l_bnScale, output);

  cudaDeviceSynchronize();
  auto kend = std::chrono::steady_clock::now();
  ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();

  cudaMemcpy(result, output, nOutput<<2, cudaMemcpyDeviceToHost);

  cudaFree(t_input);
  cudaFree(output);
  cudaFree(l_weights);
  cudaFree(ip);
  cudaFree(input);
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
  free(bias);
  free(input_);
}
