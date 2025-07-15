#ifndef _CUDA_ALGEBRA_H_
#define _CUDA_ALGEBRA_H_

#include <vector>

#include "matrix.h"

using std::vector;

template <typename T>
vector<T> gpu_lanczos_eigen(const csr_matrix<T> &matrix,
    int k, int steps);

#endif
