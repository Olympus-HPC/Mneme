#include <hip/hip_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define HIP_CHECK(cmd)                                                         \
  do {                                                                         \
    hipError_t e = cmd;                                                        \
    if (e != hipSuccess) {                                                     \
      fprintf(stderr, "HIP error: '%s' (%d) at %s:%d\n", hipGetErrorString(e), \
              e, __FILE__, __LINE__);                                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

template <typename T> __global__ void vecAdd_test(T *in, T *out, size_t size) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size)
    return;
  auto stride = gridDim.x * blockDim.x;

  for (; tid < size; tid += stride) {
    out[tid] += in[tid] + tid;
  }
}

int main(int argc, const char *argv[]) {
  if (argc < 2) {
    std::cout << "Wrong CLI, please provide:\n";
    std::cout << argv[0]
              << " '<num-elements-for-vec-add (in 10s of thousands)>'\n";
    return -1;
  }

  size_t numElements = atoi(argv[1]) * 10000;
  long *in, *out;
  long val = numElements;
  HIP_CHECK(hipMalloc((void **)&in, numElements * sizeof(long)));
  HIP_CHECK(hipMalloc((void **)&out, numElements * sizeof(long)));

  HIP_CHECK(hipMemset(in, 0, numElements * sizeof(long)));
  HIP_CHECK(hipMemset(out, 0, numElements * sizeof(long)));

  const int threads = 256;
  int num_blocks = (numElements + threads - 1) / threads;
  vecAdd_test<<<num_blocks, threads>>>(in, out, numElements);
  HIP_CHECK(hipDeviceSynchronize());

  long *h_in = new long[numElements];
  long *h_out = new long[numElements];
  HIP_CHECK(
      hipMemcpy(h_in, in, sizeof(long) * numElements, hipMemcpyDeviceToHost));
  HIP_CHECK(
      hipMemcpy(h_out, out, sizeof(long) * numElements, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());
  int ret = 0;
  for (int i = 0; i < numElements; i++) {
    if (i != h_out[i]) {
      std::cout << "Values " << i << " " << (int64_t)h_out[i] << " differ\n";
      ret = -1;
      break;
    }
  }
  if (ret == 0)
    std::cout << "Correct output\n";

  delete[] h_in;
  delete[] h_out;
  HIP_CHECK(hipFree(in));
  HIP_CHECK(hipFree(out));
  return ret;
}
