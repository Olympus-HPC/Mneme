#include <chrono>
#include <iostream>
#include <fstream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "device.h"

size_t remaining_B(int , size_t );
void preprocessing(float * , int , int );

__global__ void ker(const float * cormat, float * upper,int n)
{
  size_t idx = blockDim.x*blockIdx.x+threadIdx.x;
  if (idx < (size_t)n * n) {
    size_t i = idx/n;
    size_t j = idx%n;
    if(i<j)
    {
      size_t t = n * i - i * (i+1) / 2 + j - i - 1;
      //printf("(%lu, %lu) %lu %lu\n", i, j, t, i*n+j);
      upper[t]=cormat[j*n+i];
    }
  }
}

__global__ void ker2(const float * cormat, float * upper, int n1, int n)
{
  size_t idx = blockDim.x*blockIdx.x+threadIdx.x;
  if (idx < (size_t)n1 * n) {
    size_t i = idx/n;
    size_t j = idx%n;
    if(i<j && i<n1)
    {
      size_t t = n * i - i * (i+1) / 2 + j - i - 1;
      upper[t]=cormat[j*n1+i];
    }
  }
}


int CorMat_singlePass(float* upper_tri, float * data, int N, int L)
{
  size_t M1 = (N-1); //computing the  size of correlaion matrix
  M1 *= N;
  M1 /= 2;
  size_t total=N*N;//size of total correlation matrix

  preprocessing(data, N, L);//Preprocessing fMRI data in CPU

  cublasStatus_t stat;
  cublasHandle_t handle;
  stat = cublasCreate(&handle) ;
  if (stat != CUBLAS_STATUS_SUCCESS)
  {
    std::cout<<"Error in creating cublas handle";
    return stat;
  }

  float * dev_data; //Allocating space in GPU for storing fMRI data
  cudaMalloc ((void**)&dev_data, sizeof(float) * L * N) ;

  stat = cublasSetMatrix(N, L, sizeof(float), data, N, dev_data, N);//Copying fMRI data from CPU to GPU
  if (stat != CUBLAS_STATUS_SUCCESS) {
    std::cout<<"Error in copying data to GPU";
    return stat;
  }

  const float alpha = 1.0;
  const float beta = 0.0;

  float* dev_cormat;//allocating space in GPU for whole correlation matrix
  cudaMalloc ((void**)&dev_cormat, sizeof(float) * total) ;

  //Performing matrix multiplication (fMRI data to its transpose)
  auto start = std::chrono::steady_clock::now();
  stat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     N,N,L,
                     &alpha, dev_data, L,
                     dev_data, L, &beta,
                     dev_cormat, N);
  if (stat != CUBLAS_STATUS_SUCCESS)
  {
    std::cout<<"Error in performing multiplication";
    return stat;
  }
  cudaDeviceSynchronize(); // required for timing
  auto end = std::chrono::steady_clock::now();
  auto gemm_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  float* dev_upper;//Allocating space for extracting upper triangle part
  cudaMalloc ((void**)&dev_upper, sizeof(float) * M1) ;

  int block_size=THREAD_BLOCK_SIZE;//number of threads
  size_t grid_size=1+((total-1)/block_size);//number of blocks

  start = std::chrono::steady_clock::now();

  ker<<<grid_size,block_size>>>(dev_cormat,dev_upper,N);//performing kernel for extracting and reordering correlations from upper triangle

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  auto extract_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  cudaMemcpy(upper_tri, dev_upper, sizeof(float) * M1, cudaMemcpyDeviceToHost);//copying upper triangle correlation matrix data back to CPU
  cudaFree (dev_data);
  cudaFree (dev_cormat);
  cudaFree (dev_upper);

  stat = cublasDestroy(handle);
  if (stat != CUBLAS_STATUS_SUCCESS)
  {
    std::cout<<"Error in destroying cublas handle";
    return stat;
  }

  std::cout << "Kernel time (s)\n"
            << "GEMM: " << gemm_time * 1e-9 << ", "
            << "Extract upper triangle: " << extract_time * 1e-9 << "\n";

  return 1;
}


int CorMat_multiPass(float* upper_tri, float * data, int N, int L)
{
  int flag=1;

  preprocessing(data, N, L);//Preprocessing fMRI data

  cublasStatus_t stat;
  cublasHandle_t handle;

  float * dev_data;//initializing normalized fMRI data in GPU
  cudaMalloc ((void**)&dev_data, sizeof(float) * L * N);

  stat = cublasSetMatrix(N, L, sizeof(float), data, N, dev_data, N);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    std::cout<<"Error in copying data to GPU";
    return stat;
  }

  stat = cublasCreate(&handle) ;
  if (stat != CUBLAS_STATUS_SUCCESS)
  {
    std::cout<<"Error in creating cublas handle";
    return stat;
  }

  const float alpha = 1.0;
  const float beta = 0.0;

  const size_t free_memory = FREE_MEMORY;
  size_t available_memory = free_memory / sizeof(float) - (long)N * L;

  int block=remaining_B(N, available_memory);
  int N_prime=N;

  float* add_uper_cpu=upper_tri;
  size_t M1,temp,temp2=0,temp3=0;
  int so_far=0;
  int pak=0;
  float* dev_cormat;
  float* dev_upper;
  size_t cormat_fullsize;
  size_t gemm_time = 0, extract_time = 0;

  while(flag==1)
  {
    if(block==N_prime)//checking for the last chunk
      flag=0;

    temp = block;
    temp *= (block +1);
    temp /= 2;
    M1=N_prime;
    M1*=block;
    M1-=temp; //M1 is the size of upper triangle part of chunk

    if(pak!=0)
    {
      cudaFree (dev_upper);
      cudaFree (dev_cormat);
    }
    cormat_fullsize=block;
    cormat_fullsize*=N_prime;

    cudaMalloc ((void**)&dev_cormat, sizeof(float) * cormat_fullsize) ;
    cudaMalloc ((void**)&dev_upper, sizeof(float) * M1) ;

    pak++;

    //multiply block x L to L x N_prime = block x N_prime
    auto start = std::chrono::steady_clock::now();
    stat = cublasSgemm(handle, CUBLAS_OP_T,  CUBLAS_OP_N,
                       block, N_prime, L,
                       &alpha, dev_data+(so_far*L), L, dev_data+(so_far*L), L, &beta,
                       dev_cormat, block);

    if (stat != CUBLAS_STATUS_SUCCESS)
    {
      std::cout<<"Error in performing multiplication";
      return stat;
    }
    cudaDeviceSynchronize(); // required for timing
    auto end = std::chrono::steady_clock::now();
    gemm_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    temp2=block;
    temp2*=N_prime;

    int block_size=THREAD_BLOCK_SIZE;
    size_t grid_size=1+((temp2-1)/block_size);

    start = std::chrono::steady_clock::now();

    ker2<<<grid_size,block_size>>>(dev_cormat,dev_upper,block,N_prime);

    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    extract_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    cudaMemcpy(add_uper_cpu, dev_upper, sizeof(float) * M1, cudaMemcpyDeviceToHost);

    temp3+=M1;
    add_uper_cpu=upper_tri+temp3;
    so_far+=block;

    if(N_prime>block)
    {
      N_prime=N_prime-block;
      block=remaining_B(N_prime, available_memory);

      if(N_prime < block)//checking last chunk
        block=N_prime;
    }
  }
  cudaFree (dev_data);
  cudaFree (dev_upper);
  cudaFree (dev_cormat);

  stat = cublasDestroy(handle);
  if (stat != CUBLAS_STATUS_SUCCESS)
  {
    std::cout<<"Error in destroying cublas handle";
    return stat;
  }

  std::cout << "Total Kernel time (s)\n"
            << "GEMM: " << gemm_time * 1e-9 << ", "
            << "Extract upper triangle: " << extract_time * 1e-9 << "\n";

  return 1;
}
