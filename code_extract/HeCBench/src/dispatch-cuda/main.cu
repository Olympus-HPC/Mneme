/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <cuda.h>

#define NUM_GROUPS 1
#define GROUP_SIZE 1
#define WARMUP_RUN_COUNT 100
#define TIMING_RUN_COUNT 1000
#define TOTAL_RUN_COUNT WARMUP_RUN_COUNT + TIMING_RUN_COUNT
#define BATCH_SIZE 1000

__global__
void EmptyKernel() { }

void print_timing(std::string test, const std::array<float, TOTAL_RUN_COUNT> &results, int batch = 1)
{
  float total_us = 0.0f, mean_us = 0.0f, stddev_us = 0.0f;

  // skip warm-up runs
  auto start_iter = std::next(results.begin(), WARMUP_RUN_COUNT);
  auto end_iter = results.end();

  // mean
  std::for_each(start_iter, end_iter, [&](const float &run_ms) {
    total_us += (run_ms * 1000) / batch;
  });
  mean_us = total_us  / TIMING_RUN_COUNT;

  // stddev
  total_us = 0;
  std::for_each(start_iter, end_iter, [&](const float &run_ms) {
    float dev_us = ((run_ms * 1000) / batch) - mean_us;
    total_us += dev_us * dev_us;
  });
  stddev_us = sqrt(total_us / TIMING_RUN_COUNT);

  // display
  printf("\n %s: mean = %.1f us, stddev = %.1f us\n", test.c_str(), mean_us, stddev_us);
}

int main() {
  std::array<float, TOTAL_RUN_COUNT> results;

  //------------------------------------------------------------------------------------
  // Kernel launch enqueue rate:
  // Measure time taken to enqueue a kernel on the GPU
  //------------------------------------------------------------------------------------
  for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    EmptyKernel<<<dim3(NUM_GROUPS), dim3(GROUP_SIZE)>>>();
    auto stop = std::chrono::high_resolution_clock::now();
    results[i] = std::chrono::duration<float, std::milli>(stop - start).count();
  }
  print_timing("Enqueue rate", results);

  //------------------------------------------------------------------------------------
  // Single dispatch execution latency
  //------------------------------------------------------------------------------------
  for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    EmptyKernel<<<dim3(NUM_GROUPS), dim3(GROUP_SIZE)>>>();
    cudaDeviceSynchronize();
    auto stop = std::chrono::high_resolution_clock::now();
    results[i] = std::chrono::duration<float, std::milli>(stop - start).count();
  }
  print_timing("Single dispatch latency", results);

  //------------------------------------------------------------------------------------
  // Batch dispatch execution latency
  //------------------------------------------------------------------------------------
  for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < BATCH_SIZE; j++) {
      EmptyKernel<<<dim3(NUM_GROUPS), dim3(GROUP_SIZE)>>>();
    }
    cudaDeviceSynchronize();
    auto stop = std::chrono::high_resolution_clock::now();
    results[i] = std::chrono::duration<float, std::milli>(stop - start).count();
  }
  print_timing("Batch dispatch latency", results, BATCH_SIZE);

  return 0;
}
