#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "linear.h"
#include "kernel.h"

#define local_size TEMP_WORKGROUP_SIZE 

int cpu_offset;

void linear_regressionOMP(data_t* dataset, sum_t* result, int gpu_size, int global_size)
{
  sum_t* interns = (sum_t*)malloc(sizeof(sum_t) * local_size);

  for (int group_id = (gpu_size / local_size); group_id < (global_size / local_size); group_id++) {
    for (int loc_id = local_size - 1; loc_id >= 0; loc_id--) {
      int glob_id = loc_id + group_id * local_size;
      interns[loc_id].x = dataset[glob_id].x;
      interns[loc_id].y = dataset[glob_id].y;
      interns[loc_id].z = (dataset[glob_id].x * dataset[glob_id].y);
      interns[loc_id].w = (dataset[glob_id].x * dataset[glob_id].x);
      for (int i = (local_size / 2), old_i = local_size; i > 0; old_i = i, i /= 2) {
        if (loc_id < i) {
          interns[loc_id].x += interns[loc_id + i].x;
          interns[loc_id].y += interns[loc_id + i].y;
          interns[loc_id].z += interns[loc_id + i].z;
          interns[loc_id].w += interns[loc_id + i].w;
          if (loc_id == (i - 1) && old_i % 2 != 0) {
            interns[loc_id].x += interns[old_i - 1].x;
            interns[loc_id].y += interns[old_i - 1].y;
            interns[loc_id].z += interns[old_i - 1].z;
            interns[loc_id].w += interns[old_i - 1].w;
          }
        }
      }
      if (loc_id == 0) result[group_id] = interns[0];
    }
  }

  free(interns);
}

void rsquaredOMP(
    data_t* dataset,
    float mean,
    float2 equation,  // [a0,a1]
    rsquared_t* result,
    int gpu_size,
    int global_size)
{
  rsquared_t* dist = (rsquared_t*)malloc(sizeof(rsquared_t) * local_size);

  for (int group_id = (gpu_size / local_size); group_id < (global_size / local_size); group_id++) {
    for (int loc_id = local_size - 1; loc_id >= 0; loc_id--) {
      int glob_id = loc_id + group_id * local_size;
      dist[loc_id].x = powf((dataset[glob_id].x - mean), 2.f);
      float y_estimated = dataset[glob_id].x * equation.x + equation.y;
      dist[loc_id].y = powf((y_estimated - mean), 2.f);
      for (int i = (local_size / 2), old_i = local_size; i > 0; old_i = i, i /= 2) {
        if (loc_id < i) {
          dist[loc_id].x += dist[loc_id + i].x;
          dist[loc_id].y += dist[loc_id + i].y;
          if (loc_id == (i - 1) && old_i % 2 != 0) {
            dist[loc_id].x += dist[old_i - 1].x;
            dist[loc_id].y += dist[old_i - 1].y;
          }
        }
      }
      if (loc_id == 0) {
        result[group_id] = dist[0];
      }
    }
  }

  free(dist);
}

static
void r_squared(linear_param_t *params, data_t *dataset, sum_t *linreg, result_t *response) 
{
  float mean = linreg->y / params->size;
  float2 equation = {response->a0, response->a1};
  
  const size_t wg_count = params->wg_count;
  const size_t wg_size = params->wg_size;
  const size_t size = params->size;

  rsquared_t *results = (rsquared_t*) malloc(sizeof(rsquared_t) * wg_count);

  size_t globalWorkSize = size;
  if (size % wg_size) globalWorkSize += wg_size - (size % wg_size);

  size_t cpu_global_size = cpu_offset * globalWorkSize / 100;
  if (cpu_global_size % wg_size != 0) {
      cpu_global_size = (1 + cpu_global_size / wg_size) * wg_size;
  }

  size_t gpu_global_size = globalWorkSize - cpu_global_size;

  data_t* d_dataset = NULL;
  rsquared_t* d_result = NULL;

  if (gpu_global_size > 0) {
    /* Create data buffer */
    cudaMalloc((void**)&d_dataset, sizeof(data_t) * size);
    cudaMalloc((void**)&d_result, sizeof(rsquared_t) * wg_count);
    cudaMemcpy(d_dataset, dataset, sizeof(data_t) * size, cudaMemcpyHostToDevice);

    dim3 grids (gpu_global_size / wg_size);
    dim3 blocks (wg_size);

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < params->repeat; i++)
      rsquared<<<grids, blocks, wg_size * sizeof(rsquared_t)>>>(d_dataset, mean, equation, d_result);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    response->ktime += time;
  }

  if (cpu_offset > 0) {
    rsquaredOMP(dataset, mean, equation, results, gpu_global_size, params->size);
  }

  if (gpu_global_size > 0) {
    cudaMemcpy(results, d_result, sizeof(rsquared_t) * (wg_count * (100-cpu_offset) / 100), cudaMemcpyDeviceToHost);
  }

  rsquared_t final_result = {0.f, 0.f};

  for (size_t i = 0; i < wg_count; i++) {
    final_result.x += results[i].x;
    final_result.y += results[i].y;
  }

  response->rsquared = final_result.y / final_result.x * 100;

  free(results);
  if (gpu_global_size > 0) {
    cudaFree(d_dataset);
    cudaFree(d_result);
  }
}

void parallelized_regression(linear_param_t *params, data_t *dataset, result_t *response) 
{
  const size_t size = params->size;
  const size_t wg_size = params->wg_size;
  const size_t wg_count = params->wg_count;

  /* Data and buffers */
  sum_t *results = (sum_t *) malloc(sizeof(sum_t) * wg_count);

  size_t globalWorkSize = size;
  if (size % wg_size) globalWorkSize += wg_size - (size % wg_size);

  size_t cpu_global_size;
  cpu_global_size = cpu_offset * globalWorkSize / 100;
  if (cpu_global_size % wg_size != 0) {
    cpu_global_size = (1 + cpu_global_size / wg_size) * wg_size;
  }

  size_t gpu_global_size;
  gpu_global_size = globalWorkSize - cpu_global_size;

  data_t* d_dataset = NULL;
  sum_t* d_result = NULL;
    
  if (gpu_global_size > 0) {
    /* Create data buffer */
    cudaMalloc((void**)&d_dataset, sizeof(data_t) * size);
    cudaMalloc((void**)&d_result, sizeof(sum_t) * wg_count);
    cudaMemcpy(d_dataset, dataset, sizeof(data_t) * size, cudaMemcpyHostToDevice);

    dim3 grids (gpu_global_size / wg_size);
    dim3 blocks (wg_size);

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < params->repeat; i++)
      linear_regression<<<grids, blocks, wg_size * sizeof(sum_t)>>>(d_dataset, d_result);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    response->ktime += time;
  }

  if (cpu_offset > 0) {
    linear_regressionOMP(dataset, results, gpu_global_size, size);
  }

  if (gpu_global_size > 0) {
    cudaMemcpy(results, d_result, sizeof(sum_t) * (wg_count * (100-cpu_offset) / 100), cudaMemcpyDeviceToHost);
  }

  /* Finalize algorithm */
  sum_t final_result = {0.f, 0.f, 0.f, 0.f};

  for (size_t i = 0; i < wg_count; i++) {
    final_result.x += results[i].x;
    final_result.y += results[i].y;
    final_result.z += results[i].z;
    final_result.w += results[i].w;
  }

  double denom = (size * final_result.w - (final_result.x * final_result.x));
  response->a0 = (final_result.y * final_result.w - final_result.x * final_result.z) / denom;
  response->a1 = (size * final_result.z - final_result.x * final_result.y) / denom;
#ifdef DEBUG
  printf("%f %f %f\n", denom, response->a0, response->a1);
#endif

  /* Deallocate resources  */
  free(results);
  if (gpu_global_size > 0) {
    cudaFree(d_dataset);
    cudaFree(d_result);
  }

  r_squared(params, dataset, &final_result, response);
}
