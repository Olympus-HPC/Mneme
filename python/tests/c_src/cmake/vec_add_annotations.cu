#include <cstdlib>
#include <iostream>
#include <string>

#define CAT2(a, b) a##b
#define CAT(a, b) CAT2(a, b)

#ifdef __ENABLE_CUDA__
#include <cuda_runtime.h>
#define DEV_PREFIX cuda
#elif defined(__ENABLE_HIP__)
#include <hip/hip_runtime.h>
#define DEV_PREFIX hip
#endif

#include "mneme/MnemeAnnotation.hpp"

#define prefix(name) CAT(DEV_PREFIX, name)

template <typename T>
__global__ void vecAdd_test(T *in, T *out, size_t size) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size)
    return;
  out[tid] += in[tid] + tid;
}

int main(int argc, const char *argv[]) {
  if (argc != 2) {
    std::cerr << "Wrong CLI: Proper one is " << argv[0] << " '<num elements>'\n";
    return 1;
  }

  size_t numElements = std::atoi(argv[1]);
  double *in = nullptr;
  double *out = nullptr;

  prefix(Malloc)((void **)&in, numElements * sizeof(double));
  prefix(Malloc)((void **)&out, numElements * sizeof(double));

  prefix(Memset)(in, 0, numElements * sizeof(double));
  prefix(Memset)(out, 0, numElements * sizeof(double));

  mneme::annotate(in, mneme::Metadata{.builtin = mneme::BuiltinDType::F64,
                                      .threshold = 0.125,
                                      .threshold_kind = mneme::ThresholdKind::Absolute,
                                      .norm = mneme::Norm::L2,
                                      .tag = std::string("in_vec")});

  mneme::annotate(out, mneme::Metadata{.builtin = mneme::BuiltinDType::F64,
                                       .threshold = 0.01,
                                       .threshold_kind = mneme::ThresholdKind::Relative,
                                       .norm = mneme::Norm::Linf,
                                       .tag = std::string("out_loose")});

  vecAdd_test<<<4, 128>>>(in, out, numElements);
  prefix(DeviceSynchronize)();

  // Update metadata on the same pointer so a different dynamic launch captures
  // a different annotation state in its prologue snapshot.
  mneme::annotate(out, mneme::Metadata{.builtin = mneme::BuiltinDType::F64,
                                       .threshold = 0.005,
                                       .threshold_kind = mneme::ThresholdKind::Relative,
                                       .norm = mneme::Norm::Linf,
                                       .tag = std::string("out_tight")});

  vecAdd_test<<<8, 64>>>(in, out, numElements);
  prefix(DeviceSynchronize)();

  prefix(Free)(in);
  prefix(Free)(out);
  std::cout << "annotation test complete\n";
  return 0;
}
