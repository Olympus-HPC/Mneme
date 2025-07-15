#include <vector>
#include <iostream>
#include <chrono>
#include <cuda.h>
#include "util.h"  // graph

#define DIAMETER_SAMPLES 512

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#ifndef checkCudaErrors
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(cudaError_t err, const char *file, const int line)
{   
  if (cudaSuccess != err)
  {   
    std::cerr << "CUDA Error = " << err << ": " << cudaGetErrorString(err) << " from file "
              << file  << ", line " << line << std::endl;
  }
}
#endif

//Note: N must be a power of two
//Simple/Naive bitonic sort. We're only sorting ~512 elements one time, so performance isn't important
__device__ void bitonic_sort(int *values, const int N)
{
  unsigned int idx = threadIdx.x;

  for (int k = 2; k <= N; k <<= 1)
  {
    for (int j = k >> 1; j > 0; j = j >> 1)
    {
      while(idx < N) 
      {
        int ixj = idx^j;
        if (ixj > idx) 
        {
          if ((idx&k) == 0 && values[idx] > values[ixj]) 
          {
            //exchange(idx, ixj);
            int tmp = values[idx];
            values[idx] = values[ixj];
            values[ixj] = tmp;
          }
          if ((idx&k) != 0 && values[idx] < values[ixj]) 
          {
            //exchange(idx, ixj);
            int tmp = values[idx];
            values[idx] = values[ixj];
            values[ixj] = tmp;
          }
        }
        idx += blockDim.x;
      }
      __syncthreads();
      idx = threadIdx.x;
    }
  }
}


__global__ void bc_kernel(
  float *__restrict__ bc,
  const int *__restrict__ R,
  const int *__restrict__ C,
  const int *__restrict__ F,
  const int n,
  const int m,
  const int *__restrict__ d,
  const unsigned long long *__restrict__ sigma,
  const float *__restrict__ delta,
  const int *__restrict__ Q,
  const int *__restrict__ Q2,
  const int *__restrict__ S,
  const int *__restrict__ endpoints,
  int *__restrict__ next_source,
  const size_t pitch_d,
  const size_t pitch_sigma,
  const size_t pitch_delta,
  const size_t pitch_Q,
  const size_t pitch_Q2,
  const size_t pitch_S,
  const size_t pitch_endpoints,
  const int start,
  const int end,
  int *__restrict__ jia,
  int *__restrict__ diameters,
  const int *__restrict__ source_vertices,
  const bool approx)
{
  __shared__ int ind;
  __shared__ int i;
  __shared__ int *Q_row;
  __shared__ int *Q2_row;
  __shared__ int *S_row;
  __shared__ int *endpoints_row;

  int j = threadIdx.x;
  int *d_row = (int*)((char*)d + blockIdx.x*pitch_d);
  unsigned long long *sigma_row = (unsigned long long*)((char*)sigma + blockIdx.x*pitch_sigma);
  float *delta_row = (float*)((char*)delta + blockIdx.x*pitch_delta);
  if(j == 0)
  {
    ind = blockIdx.x + start;
    i = approx ? source_vertices[ind] : ind;
    Q_row = (int*)((char*)Q + blockIdx.x*pitch_Q);
    Q2_row = (int*)((char*)Q2 + blockIdx.x*pitch_Q2);
    S_row = (int*)((char*)S + blockIdx.x*pitch_S);
    endpoints_row = (int*)((char*)endpoints + blockIdx.x*pitch_endpoints);
    *jia = 0;
  }
  __syncthreads();

  if((ind==0) && (j < DIAMETER_SAMPLES))
  {
    diameters[j] = INT_MAX;
  }
  __syncthreads();

  while(ind < end)
  {
    //Initialization
    for(int k=threadIdx.x; k<n; k+=blockDim.x)
    {
      if(k == i) //If k is the source node...
      {
        d_row[k] = 0;
        sigma_row[k] = 1;
      }
      else
      {
        d_row[k] = INT_MAX;
        sigma_row[k] = 0;
      }  
      delta_row[k] = 0;
    }
    __syncthreads();

    //Shortest Path Calculation
    __shared__ int Q_len;
    __shared__ int Q2_len;
    __shared__ int S_len;
    __shared__ int current_depth; 
    __shared__ int endpoints_len;
    __shared__ bool sp_calc_done;

    if(j == 0)
    {
      Q_row[0] = i;
      Q_len = 1;
      Q2_len = 0;
      S_row[0] = i;
      S_len = 1;
      endpoints_row[0] = 0;
      endpoints_row[1] = 1;
      endpoints_len = 2;
      current_depth = 0;
      sp_calc_done = false;
    }
    __syncthreads();

    //Do first iteration separately since we already know the edges to traverse
    for(int r=threadIdx.x+R[i]; r<R[i+1]; r+=blockDim.x)
    {
      int w = C[r];
      //No multiple/self edges - each value of w is unique, so no need for atomics
      if(d_row[w] == INT_MAX)
      {
        d_row[w] = 1; 
        int t = atomicAdd(&Q2_len,1);
        Q2_row[t] = w;
      }
      if(d_row[w] == (d_row[i]+1))
      {
        atomicAdd(&sigma_row[w],1); 
      }
    }
    __syncthreads();

    if(Q2_len == 0)
    {
      sp_calc_done = true;
    }
    else
    {
      for(int kk=threadIdx.x; kk<Q2_len; kk+=blockDim.x)
      {
        Q_row[kk] = Q2_row[kk];
        S_row[kk+S_len] = Q2_row[kk];
      }
      __syncthreads();
      if(j == 0)
      {
        endpoints_row[endpoints_len] = endpoints_row[endpoints_len-1] + Q2_len;
        endpoints_len++;
        Q_len = Q2_len;
        S_len += Q2_len;
        Q2_len = 0;
        current_depth++;
      }
    }
    __syncthreads();

    while(!sp_calc_done)
    {
      if((*jia) && (Q_len > 512))
      {
        for(int k=threadIdx.x; k<2*m; k+=blockDim.x)
        {
          int v = F[k];
          if(d_row[v] == current_depth) 
          {
            int w = C[k];
            if(atomicCAS(&d_row[w],INT_MAX,d_row[v]+1) == INT_MAX)
            {
              int t = atomicAdd(&Q2_len,1);
              Q2_row[t] = w;
            }
            if(d_row[w] == (d_row[v]+1))
            {
              atomicAdd(&sigma_row[w],sigma_row[v]);
            }
          }  
        }
      }
      else
      {
        __shared__ int next_index;
        if(j == 0)
        {
          next_index = blockDim.x;
        }
        __syncthreads();
        int k = threadIdx.x; //Initial vertices
        while(k < Q_len)
        {
          int v = Q_row[k];
          for(int r=R[v]; r<R[v+1]; r++)
          {
            int w = C[r];
            //Use atomicCAS to prevent duplicates
            if(atomicCAS(&d_row[w],INT_MAX,d_row[v]+1) == INT_MAX)
            {
              int t = atomicAdd(&Q2_len,1);
              Q2_row[t] = w;
            }
            if(d_row[w] == (d_row[v]+1))
            {
              atomicAdd(&sigma_row[w],sigma_row[v]);
            }
          }
          k = atomicAdd(&next_index,1);
        }
      }
      __syncthreads();

      if(Q2_len == 0) //If there is no additional work found, we're done
      {
        break;
      }
      else //If there is additional work, transfer elements from Q2 to Q, reset lengths, and add vertices to the stack
      {
        for(int kk=threadIdx.x; kk<Q2_len; kk+=blockDim.x)
        {
          Q_row[kk] = Q2_row[kk];
          S_row[kk+S_len] = Q2_row[kk];
        }
        __syncthreads();
        if(j == 0)
        {
          endpoints_row[endpoints_len] = endpoints_row[endpoints_len-1] + Q2_len;
          endpoints_len++;
          Q_len = Q2_len;
          S_len += Q2_len;
          Q2_len = 0;
          current_depth++;
        }
        __syncthreads();
      }
    }

    //The elements at the end of the stack will have the largest distance from the source
    //Using the successor method, we can start from one depth earlier
    if(j == 0)
    {
      current_depth = d_row[S_row[S_len-1]] - 1;
      if(ind<DIAMETER_SAMPLES)
      {
        diameters[ind] = current_depth+1;
      }
    }
    __syncthreads();

    //Dependency Accumulation (Madduri/Ediger successor method)
    while(current_depth > 0)
    {
      int stack_iter_len = endpoints_row[current_depth+1]-endpoints_row[current_depth];
      if((*jia) && (stack_iter_len>512))
      {
        for(int kk=threadIdx.x; kk<2*m; kk+=blockDim.x)
        {
          int w = F[kk];
          if(d_row[w] == current_depth)
          {
            int v = C[kk];
            if(d_row[v] == (d_row[w]+1))
            {
              float change = (sigma_row[w]/(float)sigma_row[v])*(1.0f+delta_row[v]);
              atomicAdd(&delta_row[w],change);
            }    
          }
        }
      }
      else 
      {
        for(int kk=threadIdx.x+endpoints_row[current_depth]; kk<endpoints_row[current_depth+1]; kk+=blockDim.x)
        {
          int w = S_row[kk];
          float dsw = 0;
          float sw = (float)sigma_row[w];
          for(int z=R[w]; z<R[w+1]; z++)
          {
            int v = C[z];
            if(d_row[v] == (d_row[w]+1))
            {
              dsw += (sw/(float)sigma_row[v])*(1.0f+delta_row[v]);
            }
          }
          delta_row[w] = dsw;  
        }
      }
      __syncthreads();
      if(j == 0)
      {
        current_depth--;
      }
      __syncthreads();
    }

    for(int kk=threadIdx.x; kk<n; kk+=blockDim.x)
    {
      atomicAdd(&bc[kk],delta_row[kk]); //Would need to check that kk != i here, but delta_row[kk] is guaranteed to be 0.
    }

    if(j == 0)
    {
      ind = atomicAdd(next_source,1);
      if(approx)
      {
        i = source_vertices[ind];
      }
      else
      {
        i = ind;
      }
    }
    __syncthreads();

    if(ind == 2*DIAMETER_SAMPLES)
    {
      __shared__ int diameter_keys[DIAMETER_SAMPLES];
      for(int kk = threadIdx.x; kk<DIAMETER_SAMPLES; kk+=blockDim.x)
      {
        diameter_keys[kk] = diameters[kk];
      }
      __syncthreads();
      bitonic_sort(diameter_keys,DIAMETER_SAMPLES);
      __syncthreads();
      if(j == 0)
      {
        int log2n = 0;
        int tempn = n;
        while(tempn >>= 1)
        {
          ++log2n;
        }
        if(diameter_keys[DIAMETER_SAMPLES/2] < 4*log2n) //Use the median
        {
          *jia = 1;
        }
      }
    }
    __syncthreads();
  }
}

std::vector<float> bc_gpu(
  graph g,
  int max_threads_per_block,
  int number_of_SMs,
  program_options op,
  const std::set<int> &source_vertices)
{
  float *bc_gpu = new float[g.n];
  int next_source = number_of_SMs; 

  float *bc_d, *delta_d;
  int *d_d, *R_d, *C_d, *F_d, *Q_d, *Q2_d, *S_d, *endpoints_d, *next_source_d, *source_vertices_d;
  unsigned long long *sigma_d;
  size_t pitch_d, pitch_sigma, pitch_delta, pitch_Q, pitch_Q2, pitch_S, pitch_endpoints;
  int *jia_d, *diameters_d;

  dim3 dimGrid (number_of_SMs, 1, 1);
  dim3 dimBlock (max_threads_per_block, 1, 1); 

  //Allocate and transfer data to the GPU
  checkCudaErrors(cudaMalloc((void**)&bc_d,sizeof(float)*g.n));
  checkCudaErrors(cudaMalloc((void**)&R_d,sizeof(int)*(g.n+1)));
  checkCudaErrors(cudaMalloc((void**)&C_d,sizeof(int)*(2*g.m)));
  checkCudaErrors(cudaMalloc((void**)&F_d,sizeof(int)*(2*g.m)));

  checkCudaErrors(cudaMallocPitch((void**)&d_d,&pitch_d,sizeof(int)*g.n,dimGrid.x));
  checkCudaErrors(cudaMallocPitch((void**)&sigma_d,&pitch_sigma,sizeof(unsigned long long)*g.n,dimGrid.x));
  checkCudaErrors(cudaMallocPitch((void**)&delta_d,&pitch_delta,sizeof(float)*g.n,dimGrid.x));
  //Making Queues/Stack of size O(n) since we won't duplicate
  checkCudaErrors(cudaMallocPitch((void**)&Q_d,&pitch_Q,sizeof(int)*g.n,dimGrid.x));
  checkCudaErrors(cudaMallocPitch((void**)&Q2_d,&pitch_Q2,sizeof(int)*g.n,dimGrid.x));
  checkCudaErrors(cudaMallocPitch((void**)&S_d,&pitch_S,sizeof(int)*g.n,dimGrid.x));
  checkCudaErrors(cudaMallocPitch((void**)&endpoints_d,&pitch_endpoints,sizeof(int)*(g.n+1),dimGrid.x));

  checkCudaErrors(cudaMalloc((void**)&next_source_d,sizeof(int)));

  // source_vertices of type "std::set" has no data() method
  std::vector<int> source_vertices_h(source_vertices.size());
  std::copy(source_vertices.begin(),source_vertices.end(),source_vertices_h.begin());
  checkCudaErrors(cudaMalloc((void**)&source_vertices_d, sizeof(int) * source_vertices.size()));
  
  if(op.approx)
  {
    checkCudaErrors(cudaMemcpy(source_vertices_d, source_vertices_h.data(),
                    sizeof(int) * source_vertices.size(), cudaMemcpyHostToDevice));
  }

  checkCudaErrors(cudaMalloc((void**)&jia_d,sizeof(int)));
  checkCudaErrors(cudaMalloc((void**)&diameters_d,sizeof(int)*DIAMETER_SAMPLES));
  checkCudaErrors(cudaMemset(jia_d,0,sizeof(int)));
  checkCudaErrors(cudaMemset(diameters_d,0,sizeof(int)*DIAMETER_SAMPLES));

  checkCudaErrors(cudaMemcpy(R_d,g.R,sizeof(int)*(g.n+1),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(C_d,g.C,sizeof(int)*(2*g.m),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(F_d,g.F,sizeof(int)*(2*g.m),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(bc_d,0,sizeof(float)*g.n));
  checkCudaErrors(cudaMemcpy(next_source_d,&next_source,sizeof(int),cudaMemcpyHostToDevice));

  int end;
  bool approx;
  if(op.approx)
  { 
    end = op.k;
    approx = true;
  } else {
    end = g.n;
    approx = false;
  }

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();
  
  bc_kernel<<<dimGrid,dimBlock>>>(
      bc_d,
      R_d,
      C_d,
      F_d,
      g.n,
      g.m,
      d_d,
      sigma_d,
      delta_d,
      Q_d,
      Q2_d,
      S_d,
      endpoints_d,
      next_source_d,
      pitch_d,
      pitch_sigma,
      pitch_delta,
      pitch_Q,
      pitch_Q2,
      pitch_S,
      pitch_endpoints,
      0,
      end,
      jia_d,
      diameters_d,
      source_vertices_d,
      approx);

  cudaDeviceSynchronize();
  auto stop = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
  std::cout << "Kernel execution time " << time * 1e-9f << " (s)\n";

  // GPU result
  checkCudaErrors(cudaMemcpy(bc_gpu,bc_d,sizeof(float)*g.n,cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(bc_d));
  checkCudaErrors(cudaFree(R_d));
  checkCudaErrors(cudaFree(C_d));
  checkCudaErrors(cudaFree(F_d));
  checkCudaErrors(cudaFree(d_d));
  checkCudaErrors(cudaFree(sigma_d));
  checkCudaErrors(cudaFree(delta_d));
  checkCudaErrors(cudaFree(Q_d));
  checkCudaErrors(cudaFree(Q2_d));
  checkCudaErrors(cudaFree(S_d));
  checkCudaErrors(cudaFree(endpoints_d));
  checkCudaErrors(cudaFree(next_source_d));
  checkCudaErrors(cudaFree(jia_d));
  checkCudaErrors(cudaFree(diameters_d));
  checkCudaErrors(cudaFree(source_vertices_d));

  //Copy GPU result to a vector
  std::vector<float> bc_gpu_v(bc_gpu,bc_gpu+g.n);

  for(int i=0; i<g.n; i++)
  {
    bc_gpu_v[i] /= 2.0f; //we don't want to double count the unweighted edges
  }

  delete[] bc_gpu;
  return bc_gpu_v;
}

// query the properties of a single device for simplicity
void query_device(int &max_threads_per_block, int &number_of_SMs, program_options op)
{
  op.device = 0;
  checkCudaErrors(cudaSetDevice(op.device));
  cudaDeviceProp prop;
  checkCudaErrors(cudaGetDeviceProperties(&prop, op.device));

  std::cout << "Chosen Device: " << prop.name << std::endl;
  std::cout << "Number of Multiprocessors: " << prop.multiProcessorCount << std::endl;
  std::cout << "Size of Global Memory: " << prop.totalGlobalMem/(float)(1024*1024*1024)
            << " GB" << std::endl << std::endl;

  max_threads_per_block = prop.maxThreadsPerBlock;
  number_of_SMs = prop.multiProcessorCount;
}

