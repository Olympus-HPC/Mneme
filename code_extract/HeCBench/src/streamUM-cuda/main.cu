/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This sample implements a simple task consumer using threads and streams
 * with all data in Unified Memory, and tasks consumed by both host and device
 */

// system includes
#include <chrono>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <stdlib.h>

// cuBLAS
#include <cublas_v2.h>

// utilities
#include "reference.h"

cudaError_t checkCudaErrors(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
  }
#endif
  return result;
}

cublasStatus_t checkCublasErrors(cublasStatus_t result) {
#if defined(DEBUG) || defined(_DEBUG)
  if (result != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUBLAS Runtime Error: %s\n", cublasGetStatusString(result));
  }
#endif
  return result;
}

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
// SRAND48 and DRAND48 don't exist on windows, but these are the equivalent
// functions
void srand48(long seed) { srand((unsigned int)seed); }
double drand48() { return double(rand()) / RAND_MAX; }
#endif

// simple task
template <typename T>
struct Task {
  unsigned int size, id;
  T *data;
  T *result;
  T *vector;

  Task() : size(0), id(0), data(NULL), result(NULL), vector(NULL){};

  void deallocate() {
    checkCudaErrors(cudaFree(data));
    checkCudaErrors(cudaFree(result));
    checkCudaErrors(cudaFree(vector));
  }

  void allocate(const unsigned int s, const unsigned int unique_id) {
    // allocate unified memory outside of constructor
    id = unique_id;
    size = s;
    checkCudaErrors(cudaMallocManaged(&data, sizeof(T) * size * size));
    checkCudaErrors(cudaMallocManaged(&result, sizeof(T) * size));
    checkCudaErrors(cudaMallocManaged(&vector, sizeof(T) * size));

    // populate data with random elements
    for (unsigned int i = 0; i < size * size; i++) {
      data[i] = drand48();
    }

    for (unsigned int i = 0; i < size; i++) {
      result[i] = 0.;
      vector[i] = drand48();
    }
  }
};

// execute a single task on either host or device depending on size
template <typename T>
void execute(Task<T> &t, cublasHandle_t *handle, cudaStream_t *stream,
             int tid) {
  if (t.size < 100) {
    // perform on host
    //printf("Task [%d], thread [%d] executing on host (%d)\n", t.id, tid, t.size);

    // call the host operation
    gemv(t.size, t.size, (T)1.0, t.data, t.vector, (T)0.0, t.result);
  } else {
    // perform on device
    //printf("Task [%d], thread [%d] executing on device (%d)\n", t.id, tid, t.size);
    T one = 1.0;
    T zero = 0.0;

    // attach managed memory to my stream
    checkCublasErrors(cublasSetStream(handle[tid + 1], stream[tid + 1]));
    // call the device operation
    checkCublasErrors(cublasDgemv(handle[tid + 1], CUBLAS_OP_N, t.size, t.size,
                                &one, t.data, t.size, t.vector, 1, &zero,
                                t.result, 1));
  }
}

// populate a list of tasks with random sizes
template <typename T>
void initialise_tasks(std::vector<Task<T> > &TaskList) {
  for (unsigned int i = 0; i < TaskList.size(); i++) {
    // generate random size
    int size = std::max((int)(drand48() * 1000.0), 64);
    TaskList[i].allocate(size, i);
  }
}

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: %s <number of host threads> <number of tasks> <verify>\n", argv[0]);
    return 1;
  }
  const int nthreads = atoi(argv[1]);
  const unsigned int N = atoi(argv[2]);
  const int verify = atoi(argv[3]);

  /*
  // set device
  cudaDeviceProp device_prop;
  const int dev_id = 0;
  checkCudaErrors(cudaGetDeviceProperties(&device_prop, dev_id));

  if (!device_prop.managedMemory) {
    // This samples requires being run on a device that supports Unified Memory
    fprintf(stderr, "Unified Memory not supported on this device\n");

    exit(0);
  }

  if (device_prop.computeMode == cudaComputeModeProhibited) {
    // This sample requires being run with a default or process exclusive mode
    fprintf(stderr,
            "This sample requires a device in either default or process "
            "exclusive mode\n");

    exit(0);
  }
  */

  // randomise task sizes
  srand48(48);

  // number of streams = number of threads
  cudaStream_t *streams = new cudaStream_t[nthreads + 1];
  cublasHandle_t *handles = new cublasHandle_t[nthreads + 1];

  for (int i = 0; i < nthreads + 1; i++) {
    checkCudaErrors(cudaStreamCreate(&streams[i]));
    checkCublasErrors(cublasCreate(&handles[i]));
  }

  // create list of N tasks
  std::vector<Task<double> > TaskList(N);
  initialise_tasks(TaskList);

  printf("Executing tasks on host / device\n");

// run through all tasks using threads and streams
  omp_set_num_threads(nthreads);

  //checkCudaErrors(cudaSetDevice(dev_id));

  auto start = std::chrono::steady_clock::now();

  #pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < TaskList.size(); i++) {
    int tid = omp_get_thread_num();
    execute(TaskList[i], handles, streams, tid);
  }

  cudaDeviceSynchronize();

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Task execution time : %f (s)\n", (time * 1e-9f));

  // Verify the device results
  if (verify) {
    check(TaskList);
  }

  for (size_t i = 0; i < TaskList.size(); i++) {
    TaskList[i].deallocate();
  }

  for (int i = 0; i < nthreads + 1; i++) {
    cudaStreamDestroy(streams[i]);
    cublasDestroy(handles[i]);
  }
  delete[] streams;
  delete[] handles;

  return 0;
}
