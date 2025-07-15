#include <cmath>
#pragma once

namespace SNICIT_BEY{
__global__ void dense_input(
  const float* Y0,
  const float* weight,
  const float* bias,
  const int M, const int N, const int K,
  float* Y1
) {
    extern  __shared__ float shRow[];
    if (threadIdx.y == 0) {
        shRow[threadIdx.x] = bias[threadIdx.x];
    }

    __syncthreads();
    // if (threadIdx.x+threadIdx.y == 0 && blockIdx.x == 0) {
    //     printf("blockDim.x%d blockDim.y%d", blockDim.x, blockDim.y);
    // }
    for (int i = threadIdx.y; i < N; i += blockDim.y) {
        float valY = Y0[blockIdx.x * N + i];
        if(valY == 0) {
            continue;
        }
        float valW = weight[i*K+threadIdx.x];
        atomicAdd(&shRow[threadIdx.x], valY * valW);
    }
    __syncthreads();
    if (threadIdx.y == 0) {
        Y1[blockIdx.x * K+threadIdx.x] = min(float(1.0), max(float(0), shRow[threadIdx.x]));
    }
}


__global__ void sparse_hidden(
    
  const float* Y0,
  const int* roffW,
  const int* colsW,
  const float* valsW,
  const float* bias,
  const int M, const int N, const int K,
  float* Y1
) {
    // (8, 128)
    extern  __shared__ float shRow[];
    int tid = threadIdx.x + threadIdx.y*blockDim.x;
    if (tid < K) {
        shRow[tid] = bias[tid]; 
    }
    __syncthreads();

    for (int i = threadIdx.y; i < N; i += blockDim.y) {
        float valY = Y0[blockIdx.x * N + i];
        if(valY == 0) {
            continue;
        }

        int begOffW = roffW[i] + threadIdx.x;
        int endOffW = roffW[i + 1];
        for(int k = begOffW; k < endOffW; k += blockDim.x) { // += blockDim.x
            int colW = colsW[k];
            float valW = valsW[k];
            atomicAdd(&shRow[colW], valY * valW);
        }
    }
    __syncthreads();
    if (tid < K) {
        Y1[blockIdx.x * K+tid] = min(float(1.0), max(float(0), shRow[tid]));
    }
}

__global__ void dense_output(
  const float* Y0,
  const float* weight,
  const float* bias,
  const int M, const int N, const int K,
  float* Y1
) {
    extern  __shared__ float shRow[];
    if (threadIdx.y == 0) {
        shRow[threadIdx.x] = bias[threadIdx.x];
    }

    __syncthreads();
    // if (threadIdx.x+threadIdx.y == 0 && blockIdx.x == 0) {
    //     printf("blockDim.x%d blockDim.y%d", blockDim.x, blockDim.y);
    // }
    for (int i = threadIdx.y; i < N; i += blockDim.y) {
        float valY = Y0[blockIdx.x * N + i];
        if(valY == 0) {
            continue;
        }
        float valW = weight[i*K+threadIdx.x];
        atomicAdd(&shRow[threadIdx.x], valY * valW);
    }
    __syncthreads();
    if (threadIdx.y == 0) {
        Y1[blockIdx.x * K+threadIdx.x] = shRow[threadIdx.x];
    }
}

__global__ void check_acc(
    float* Y, int num_classes, int num_input, int* label, int *cnt
) {
    extern  __shared__ int shcnt[];
    if (threadIdx.x == 0)
        shcnt[0] = 0;
    __syncthreads(); 
    for (int i = threadIdx.x; i < num_input; i += blockDim.x) {
        int argmax = 0;
        float tmpmax = -10000.0;
        for (int j = 0; j < num_classes; j++) {
            if (Y[i*num_classes+j] > tmpmax) {
                argmax = j;
                tmpmax = Y[i*num_classes+j];
            }
        }
        // printf("label[i]=%d argmax=%d, tmpmax = %f\n", label[i], argmax, tmpmax);
        if (argmax == label[i])
            atomicAdd(&shcnt[0], 1);
    }
    // printf("shcnt[0]=%d", shcnt[0]);
    
    __syncthreads();
    if (threadIdx.x == 0)
        cnt[0] = shcnt[0];
}

}
