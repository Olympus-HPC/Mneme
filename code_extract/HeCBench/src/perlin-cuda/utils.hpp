#pragma once

#include <iostream>
#include <algorithm>
#include <cmath>
#include <cuda.h>

const uint64_t WIN_WIDTH  = 61440;
const uint64_t WIN_HEIGHT = 34560;

#define CHECK(x) \
	do { \
		if (x != cudaSuccess) { \
			std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(x) << std::endl; \
			std::exit(1); \
		} \
	} while (false)

#define LIN(x, y, w) (y * w + x)

#define CUID(x) (blockDim.x * blockIdx.x + threadIdx.x)

template<typename T>
constexpr size_t nThreads(T n, size_t max = 1024) {
  return std::min(static_cast<size_t>(max), n);
}

template<typename T>
constexpr size_t nBlocks(T n, size_t max = 1024) {
  return static_cast<size_t>(std::ceil(n / nThreads(n, max)));
}

inline std::ostream& operator<<(std::ostream& o, dim3 d) {
  o << "(" << d.x << "," << d.y << "," << d.z << ")";
  return o;
}
