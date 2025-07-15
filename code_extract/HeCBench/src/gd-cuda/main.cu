#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda.h>
#include "utils.h"

__global__ void 
L2_norm(const float *x, float* l2_norm, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(l2_norm, x[i]*x[i]);
}

__global__ void
compute (
    const float * __restrict__ x, 
          float * __restrict__ grad, 
    const int   * __restrict__ A_row_ptr,
    const int   * __restrict__ A_col_index,
    const float * __restrict__ A_value, 
    const int   * __restrict__ A_y_label,
    float * __restrict__ total_obj_val,
    int   * __restrict__ correct,
    int m ) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < m) {
    // Simple sparse matrix multiply x' = A * x
    float xp = 0.f;
    for( int j = A_row_ptr[i]; j < A_row_ptr[i+1]; ++j){
      xp += A_value[j] * x[A_col_index[j]];
    }

    // compute objective 
    float v = logf(1.f + expf(-xp * A_y_label[i]));
    atomicAdd(total_obj_val, v);

    // compute errors
    float prediction = 1.f / (1.f + expf(-xp));
    int t = (prediction >= 0.5f) ? 1 : -1;
    if (A_y_label[i] == t) atomicAdd(correct, 1);

    // compute gradient at x
    float accum = expf(-A_y_label[i] * xp);
    accum = accum / (1.f + accum);
    for(int j = A_row_ptr[i]; j < A_row_ptr[i+1]; ++j){
      float temp = -accum * A_value[j] * A_y_label[i];
      atomicAdd(&grad[A_col_index[j]], temp);
    }
  }
}

__global__ void
update(float * __restrict__ x,
       const float * __restrict__ grad,
       int m, int n, float lambda, float alpha) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float g = grad[i] / (float)m + lambda * x[i]; 
    x[i] = x[i] - alpha * g;
  }
}

int main(int argc, const char *argv[]) {
  if (argc != 5) {
    printf("Usage: %s <path to file> <lambda> <alpha> <repeat>\n", argv[0]);
    return 1;
  }
  const std::string file_path = argv[1]; 
  const float lambda = atof(argv[2]);
  const float alpha = atof(argv[3]);
  const int iters = atof(argv[4]);

  //store the problem data in variable A and the data is going to be normalized
  Classification_Data_CRS A;
  get_CRSM_from_svm(A, file_path);

  const int m = A.m; // observations
  const int n = A.n; // features

  std::vector<float> x(n, 0.f);
  std::vector<float> grad (n);

  float *d_x;
  cudaMalloc((void**)&d_x, n * sizeof(float));
  cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  float *d_grad; 
  cudaMalloc((void**)&d_grad, n * sizeof(float));

  float *d_total_obj_val;
  cudaMalloc((void**)&d_total_obj_val, sizeof(float));

  float *d_l2_norm;
  cudaMalloc((void**)&d_l2_norm, sizeof(float));

  int *d_correct;
  cudaMalloc((void**)&d_correct, sizeof(int));

  int *d_row_ptr;
  cudaMalloc((void**)&d_row_ptr, A.row_ptr.size() * sizeof(int));
  cudaMemcpy(d_row_ptr, A.row_ptr.data(), 
             A.row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice);

  int *d_col_index;
  cudaMalloc((void**)&d_col_index, A.col_index.size() * sizeof(int));
  cudaMemcpy(d_col_index, A.col_index.data(), 
             A.col_index.size() * sizeof(int), cudaMemcpyHostToDevice);

  float *d_value;
  cudaMalloc((void**)&d_value, A.values.size() * sizeof(float));
  cudaMemcpy(d_value, A.values.data(), 
             A.values.size() * sizeof(float), cudaMemcpyHostToDevice);

  int *d_y_label;
  cudaMalloc((void**)&d_y_label, A.y_label.size() * sizeof(int));
  cudaMemcpy(d_y_label, A.y_label.data(), 
             A.y_label.size() * sizeof(int), cudaMemcpyHostToDevice);

  dim3 grid ((m+255)/256);
  dim3 block (256);

  dim3 grid2 ((n+255)/256);
  dim3 block2 (256);

  float obj_val = 0.f;
  float train_error = 0.f;

  cudaDeviceSynchronize();
  long long train_start = get_time();

  for (int k = 0; k < iters; k++) {

    // reset the training status
    float total_obj_val = 0.f;
    float l2_norm = 0.f;
    int correct = 0;

    cudaMemcpy(d_total_obj_val, &total_obj_val, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_l2_norm, &l2_norm, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_correct, &correct, sizeof(int), cudaMemcpyHostToDevice);

    //reset gradient vector
    std::fill(grad.begin(), grad.end(), 0.f);

    cudaMemcpy(d_grad, grad.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // compute the total objective, correct rate, and gradient
    compute<<<grid, block>>>(
        d_x, 
        d_grad, 
        d_row_ptr,
        d_col_index,
        d_value,
        d_y_label,
        d_total_obj_val,
        d_correct,
        m
      ); 

    // display training status for verification
    L2_norm<<<grid2, block2>>>(d_x, d_l2_norm, n);

    cudaMemcpy(&total_obj_val, d_total_obj_val, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&l2_norm, d_l2_norm, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost);

    obj_val = total_obj_val / (float)m + 0.5f * lambda * l2_norm;
    train_error = 1.f-(correct/(float)m); 

    // update x (gradient does not need to be updated)
    update<<<grid2, block2>>>(d_x, d_grad, m, n, lambda, alpha);
  }

  cudaDeviceSynchronize();
  long long train_end = get_time();
  printf("Training time takes %lf (s) for %d iterations\n\n",
         (train_end - train_start) * 1e-6, iters);

  // After 100 iterations, the expected obj_val and train_error are 0.3358405828 and 0.07433331013
  printf("object value = %f train_error = %f\n", obj_val, train_error);

  cudaFree(d_row_ptr);
  cudaFree(d_col_index);
  cudaFree(d_value);
  cudaFree(d_y_label);
  cudaFree(d_x);
  cudaFree(d_grad);
  cudaFree(d_total_obj_val);
  cudaFree(d_l2_norm);
  cudaFree(d_correct);

  return 0; 
}
