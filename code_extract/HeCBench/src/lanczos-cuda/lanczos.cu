#include <cassert>
#include <iostream>
#include <cuda.h>
#include <unistd.h>
#include <cmath>
#include <utility>

#include "eigen.h"
#include "matrix.h"
#include "lanczos.h"
#include "cycle_timer.h"

#define THREADS_PER_BLOCK 256

using std::cout;
using std::endl;
using std::vector;

/**
 * @brief   Cuda kernel function for vector dot product.
 *
 * @param   N   The vector size.
 * @param   x   The first input vector.
 * @param   y   The second input vector.
 * @param   z   The temp sum per block.
 */
template <typename T>
__global__ void dot_product_kernel(const int N, const T *x, const T *y,
    T *z) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ T result[THREADS_PER_BLOCK];

    if (index < N) {
        result[threadIdx.x] = x[index] * y[index];
    } else {
        result[threadIdx.x] = 0;
    }
    __syncthreads();

    int half = THREADS_PER_BLOCK / 2;
    while (half > 0) {
        if (threadIdx.x < half) {
            result[threadIdx.x] += result[threadIdx.x + half];
        }
        __syncthreads();
        half /= 2;
    }

    if (threadIdx.x == 0) {
        z[blockIdx.x] = result[0];
    }
}

/**
 * @brief   Cuda kernel function for vector multiply in place.
 *
 * @param   N   The vector size.
 * @param   x   The input vector.
 * @param   k   The value to multiply.
 */
template <typename T>
__global__ void multiply_inplace_kernel(const int N, T *x, const T k) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N) {
        x[index] = x[index] * k;
    }
}



/**
 * @brief   Cuda kernel function for vector saxpy in place(y += a * x).
 *
 * @param   N   The vector size.
 * @param   y   The output vector.
 * @param   x   The input vector.
 * @param   a   The value to multiply.
 */
template <typename T>
__global__ void saxpy_inplace_kernel(const int N, T *y, const T *x,
    const T a) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N) {
        y[index] += a * x[index];
    }
}


/**
 * @brief   Cuda kernel function for warp sparse matrix multiplication.
 *
 * @param   group_size  The number of threads used to calculate one row.
 * @param   rows        The row number of the matrix.
 * @param   begin_row   The row to begin from in this kernel launch.
 * @param   row_ptr     Row pointers in the CSR matrix.
 * @param   col_ind     Column indexes in the CSR matrix.
 * @param   values      Data values in the CSR matrix.
 * @param   x           The input vector x to multiply.
 * @param   y           The output vector y.
 */
template <typename T>
__global__ void warp_multiply_kernel(const int group_size, const int rows,
    const int begin_row, const int *row_ptr, const int *col_ind,
    const T *values, const T *x, T *y) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int r = index / group_size + begin_row;
    int lane = index % group_size;
    __shared__ volatile T result[THREADS_PER_BLOCK];

    result[threadIdx.x] = 0;
    if (r < rows) {
        int start = row_ptr[r];
        int end = row_ptr[r + 1];
        for (int i = start + lane; i < end; i+= group_size) {
            result[threadIdx.x] += values[i] * x[col_ind[i]];
        }
        // Threads in a warp are synchronized, so we can do this
        int half = group_size / 2;
        while (half > 0) {
            if (lane < half) {
                result[threadIdx.x] += result[threadIdx.x + half];
            }
            half /= 2;
        }
        if (lane == 0) {
            y[r] = result[threadIdx.x];
        }
    }
}

template <typename T>
T device_dot_product(int n, const T *device_x, const T *device_y, T *device_scratch) {
    const int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Run kernel
    dot_product_kernel<T><<<blocks, THREADS_PER_BLOCK>>>(
        n, device_x, device_y, device_scratch);

    // Transfer result back from device to host
    T host_scratch[blocks];
    T result(0);
    cudaMemcpy(host_scratch, device_scratch, sizeof(T) * blocks, cudaMemcpyDeviceToHost);
    for (int i = 0; i < blocks; i++) {
        result += host_scratch[i];
    }
    return result;
}



/**
 * @brief   Caller function for naive Lanczos algorithm in CUDA.
 *
 * @param   m       The matrix to do operations on.
 * @param   v       The initial vector with norm 1.
 * @param   steps   The iteration times for lanczos algorithm.
 *
 * @return  The tridiagonal matrix result of lanczos algorithm.
 */
template <typename T>
symm_tridiag_matrix<T> gpu_lanczos(const csr_matrix<T> &m,
    const vector<T> &v, const int steps) {
    symm_tridiag_matrix<T> result(steps + 1);

    int rows = m.row_size();
    int cols = m.col_size();
    int nonzeros = m.nonzeros();
    const int blocks = (rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    assert(rows == cols);
    assert(cols == v.size());

    // Malloc device space
    int *row_ptr, *col_ind;
    T *values, *x, *x_prev, *y, *scratch;
    cudaMalloc(&row_ptr, sizeof(int) * (rows + 1));
    cudaMalloc(&col_ind, sizeof(int) * nonzeros);
    cudaMalloc(&values, sizeof(T) * nonzeros);
    cudaMalloc(&x, sizeof(T) * cols);
    cudaMalloc(&x_prev, sizeof(T) * cols);
    cudaMalloc(&y, sizeof(T) * cols);
    cudaMalloc(&scratch, sizeof(T) * blocks);

    // Transfer data from host to device
    cudaMemcpy(row_ptr, m.row_ptr_data(), sizeof(int) * (rows + 1),
        cudaMemcpyHostToDevice);
    cudaMemcpy(col_ind, m.col_ind_data(), sizeof(int) * nonzeros,
        cudaMemcpyHostToDevice);
    cudaMemcpy(values, m.values_data(), sizeof(T) * nonzeros,
        cudaMemcpyHostToDevice);
    cudaMemcpy(x, v.data(), sizeof(T) * cols, cudaMemcpyHostToDevice);

    const int row_nonzeros = nonzeros / rows;
    int group_size = row_nonzeros > 16 ? 32 : 16;
    group_size = row_nonzeros > 8 ? group_size : 8;
    group_size = row_nonzeros > 4 ? group_size : 4;
    group_size = row_nonzeros > 2 ? group_size : 2;
    const int groups_per_block = THREADS_PER_BLOCK / group_size;
    const int multiply_blocks = (rows + groups_per_block - 1) / groups_per_block;
    // Run kernel and the values of alpha and beta are saved in the 'result' array
    double start_time = cycle_timer::current_seconds();
    for (int i = 0; i < steps; i++) {
        // y_i = M*x_i
        warp_multiply_kernel<T><<<multiply_blocks, THREADS_PER_BLOCK>>>(group_size,
            rows, 0, row_ptr, col_ind, values, x, y);

        // alpha_i <- y_i*x_i
        T product = device_dot_product(rows, x, y, scratch);

        result.alpha(i) = product;

        // y_i <- y_i - alpha_i*x_i - beta_i*x_(i-1)
        saxpy_inplace_kernel<T><<<blocks, THREADS_PER_BLOCK>>>(rows, y, x, -product);

        if (i > 0) {
            saxpy_inplace_kernel<T><<<blocks, THREADS_PER_BLOCK>>>(rows,
                y, x_prev, -result.beta(i - 1));
        }
        std::swap(x, x_prev);

        // beta_(i+1) <- ||y_i||
        result.beta(i) = T(sqrt(device_dot_product(rows, y, y, scratch)));

        // x_(i+1) <- y_i / beta_(i+1)
        multiply_inplace_kernel<T><<<blocks, THREADS_PER_BLOCK>>>(rows, y, 1 / result.beta(i));
        std::swap(x, y);
    }
    cudaDeviceSynchronize();
    double end_time = cycle_timer::current_seconds();
    cout << "GPU Lanczos iterations: " << steps << endl;
    cout << "GPU Lanczos time: " << end_time - start_time << " sec" << endl;

    // Release device space
    cudaFree(row_ptr);
    cudaFree(col_ind);
    cudaFree(values);
    cudaFree(x);
    cudaFree(x_prev);
    cudaFree(y);
    cudaFree(scratch);

    result.resize(steps);
    return result;
}

/**
 * @brief   Lanczos algorithm for eigendecomposition in CUDA.
 * 
 * @param   matrix  CSR matrix to decompose
 * @param   k       number of largest eigenvalues to compute
 * @param   steps   maximum steps for the iteration
 * @tparam  T       matrix element data type
 * @return  list of eigenvalues
 */
template <typename T>
vector<T> gpu_lanczos_eigen(const csr_matrix<T> &matrix, int k, int steps) {
    int cols = matrix.col_size();
    assert(cols > 0);
    vector<T> v(cols, 0);
    v[0] = 1;
    symm_tridiag_matrix<T> tridiag = gpu_lanczos(matrix, v, steps);
    return lanczos_no_spurious(tridiag, k);
}


template vector<float> gpu_lanczos_eigen(const csr_matrix<float> &matrix, int k, int steps);
template vector<double> gpu_lanczos_eigen(const csr_matrix<double> &matrix, int k, int steps);
