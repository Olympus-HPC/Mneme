// Copyright (c) 2019-2020, NVIDIA CORPORATION.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda.h>

///////////////////////////////////////////////////////////////////////////////
//                                SOSFILT                                    //
///////////////////////////////////////////////////////////////////////////////
#define MAX_THREADS 256
#define THREADS 32  
#define sos_width  6   // https://www.mathworks.com/help/signal/ref/sosfilt.html 

template<typename T>
__global__ void sosfilt( 
    const int n_signals,
    const int n_samples,
    const int n_sections,
    const int zi_width,
    const T *__restrict__ sos,
    const T *__restrict__ zi,
          T *__restrict__ x_in)
{
  extern __shared__ char smem[];
  T *s_out = reinterpret_cast<T *>( smem );
  T *s_zi = reinterpret_cast<T *>( &s_out[n_sections] ) ;
  T *s_sos = reinterpret_cast<T *>( &s_zi[n_sections * zi_width] ) ;

  // dim3 blocksPerGrid (1, blocks);
  // dim3 threadsPerBlock (256, 1);
  const int tx = static_cast<int>( threadIdx.x ) ;
  const int ty = static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) ;

  // Reset shared memory
  s_out[tx] = 0;

  // Load zi
  for ( int i = 0; i < zi_width; i++ ) {
    s_zi[tx * zi_width + i] = zi[ty * n_sections * zi_width + tx * zi_width + i];
  }

  // Load SOS
#pragma unroll 
  for ( int i = 0; i < sos_width; i++ ) {
    s_sos[tx * sos_width + i] = sos[tx * sos_width + i];
  }

  __syncthreads( );

  const int load_size = n_sections - 1 ;
  const int unload_size = n_samples - load_size ;

  T temp;
  T x_n;

  if ( ty < n_signals ) {
    // Loading phase
    for ( int n = 0; n < load_size; n++ ) {
      if ( tx == 0 ) {
        x_n = x_in[ty * n_samples + n];
      } else {
        x_n = s_out[tx - 1];
      }

      // Use direct II transposed structure
      temp = s_sos[tx * sos_width + 0] * x_n + s_zi[tx * zi_width + 0];

      s_zi[tx * zi_width + 0] =
        s_sos[tx * sos_width + 1] * x_n - s_sos[tx * sos_width + 4] * temp + s_zi[tx * zi_width + 1];

      s_zi[tx * zi_width + 1] = s_sos[tx * sos_width + 2] * x_n - s_sos[tx * sos_width + 5] * temp;

      s_out[tx] = temp;

      __syncthreads( );
    }

    // Processing phase
    for ( int n = load_size; n < n_samples; n++ ) {
      if ( tx == 0 ) {
        x_n = x_in[ty * n_samples + n];
      } else {
        x_n = s_out[tx - 1];
      }

      // Use direct II transposed structure
      temp = s_sos[tx * sos_width + 0] * x_n + s_zi[tx * zi_width + 0];

      s_zi[tx * zi_width + 0] =
        s_sos[tx * sos_width + 1] * x_n - s_sos[tx * sos_width + 4] * temp + s_zi[tx * zi_width + 1];

      s_zi[tx * zi_width + 1] = s_sos[tx * sos_width + 2] * x_n - s_sos[tx * sos_width + 5] * temp;

      if ( tx < load_size ) {
        s_out[tx] = temp;
      } else {
        x_in[ty * n_samples + ( n - load_size )] = temp;
      }

      __syncthreads( );
    }

    // Unloading phase
    for ( int n = 0; n < n_sections; n++ ) {
      // retire threads that are less than n
      if ( tx > n ) {
        x_n = s_out[tx - 1];

        // Use direct II transposed structure
        temp = s_sos[tx * sos_width + 0] * x_n + s_zi[tx * zi_width + 0];

        s_zi[tx * zi_width + 0] =
          s_sos[tx * sos_width + 1] * x_n - s_sos[tx * sos_width + 4] * temp + s_zi[tx * zi_width + 1];

        s_zi[tx * zi_width + 1] = s_sos[tx * sos_width + 2] * x_n - s_sos[tx * sos_width + 5] * temp;

        if ( tx < load_size ) {
          s_out[tx] = temp;
        } else {
          x_in[ty * n_samples + ( n + unload_size )] = temp;
        }
      }
      __syncthreads( );
    }
  }
}

template <typename T>
void filtering (const int repeat,
                const int n_signals, const int n_samples,
                const int n_sections, const int zi_width)
{
  // the number of second-order sections must be less than max threads per block
  assert(MAX_THREADS >= n_sections);

  // The number of samples must be greater than or equal to the number of sections
  assert(n_samples >= n_sections);

  // randomize input data
  srand(2);

  const int blocks = n_signals;

  dim3 blocksPerGrid (1, blocks);
  dim3 threadsPerBlock (THREADS, 1);

  // Second-order section digital filter
  const int sos_size = n_sections * sos_width ;

  T* sos = (T*) malloc (sizeof(T) * sos_size);
  for (int i = 0; i < n_sections; i++)
    for (int j = 0; j < sos_width; j++)
      sos[i*sos_width+j] = (T)1 ; // for test 

  T* d_sos;
  cudaMalloc((void**)&d_sos, sizeof(T) * sos_size);
  cudaMemcpy(d_sos, sos, sizeof(T) * sos_size, cudaMemcpyHostToDevice);

  // initial  conditions
  const int z_size = (n_sections + 1) * blocks * zi_width;
  T* zi = (T*) malloc (sizeof(T) * z_size);
  for (int i = 0; i < z_size; i++) zi[i] = (T)1; // for test

  T* d_zi;
  cudaMalloc((void**)&d_zi, sizeof(T) * z_size);
  cudaMemcpy(d_zi, zi, sizeof(T) * z_size, cudaMemcpyHostToDevice);

  // input signals
  const int x_size = n_signals * n_samples;
  T* x = (T*) malloc (sizeof(T) * x_size);
  for (int i = 0; i < n_signals; i++) 
    for (int j = 0; j < n_samples; j++) 
      x[i*n_samples+j] = (T)std::sin(2*3.14*(i+1+j));

  T* d_x;
  cudaMalloc((void**)&d_x, sizeof(T) * x_size);
  cudaMemcpy(d_x, x, sizeof(T) * x_size, cudaMemcpyHostToDevice);

  const int out_size = n_sections;
  const int shared_mem = (out_size + z_size + sos_size) * sizeof(T); 

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++)
    sosfilt<T><<<blocksPerGrid, threadsPerBlock, shared_mem, 0>>>(n_signals, 
      n_samples,
      n_sections,
      zi_width,
      d_sos,
      d_zi,
      d_x);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %lf (s)\n", time * 1e-9 / repeat);

  cudaMemcpy(x, d_x, sizeof(T) * x_size, cudaMemcpyDeviceToHost);
#ifdef DEBUG
  for (int i = 0; i < n_signals; i++) { 
    for (int j = 0; j < n_samples; j++) 
      printf("%.2f ", x[i*n_samples+j]);
    printf("\n");
  }
#endif

  free(x);
  free(sos);
  free(zi);
  cudaFree(d_x);
  cudaFree(d_sos);
  cudaFree(d_zi);
}

int main(int argc, char** argv) 
{
  if (argc != 2) 
  {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const int numSections = THREADS; 

#ifdef DEBUG
  const int numSignals = 2; 
  const int numSamples = THREADS+1;
#else
  // failed to launch the double-precision kernel when numSignals = 16 on a P100 GPU
  const int numSignals = 8;  
  const int numSamples = 100000;
#endif

  const int zi_width = 2;
  filtering<float> (repeat, numSignals, numSamples, numSections, zi_width);
  filtering<double> (repeat, numSignals, numSamples, numSections, zi_width);
  return 0;
}
