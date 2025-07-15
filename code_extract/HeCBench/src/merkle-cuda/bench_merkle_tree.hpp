#pragma once
#include <cuda.h>

// Returns elapased time of the function in nanosecond
uint64_t
benchmark_merklize_approach_1(const size_t leaf_count,
                              const size_t wg_size);
