// clang-format off
// RUN: rm -rf *.json Recorded*.bc DeviceState*.mneme
// RUN: LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_LOG_LEVEL=debug MNEME_PAGE_SIZE=%PG ./test_rr_device_mem%ext | FileCheck %s --check-prefixes=CHECK
// RUN: %RR | FileCheck %s --check-prefix=CHECK-RR
// RUN: rm -rf *.json Recorded*.bc DeviceState*.mneme
// clang-format on

#include <climits>
#include <cstdio>

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeUtils.hpp"
using namespace mneme;

#ifdef MNEME_ENABLE_HIP
using MnemeDeviceRT = DeviceTraits<DeviceVendors::HIP>;
#define SYMBOL(x) HIP_SYMBOL(x)
#endif

// CHECK-RR: DemangledName: test(A, float*, int)
// CHECK-RR: NumModules: 1
// CHECK-RR: NumInstances: 1
// CHECK-RR: BlockDims:(256, 1, 1)
// CHECK-RR: GridDims:(16, 1, 1)
// CHECK-RR: Results Match

struct A {
  int x, y, z;
};

__global__ void test(A data, float *out, int N) {
  auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  auto stride = gridDim.x * blockDim.x;

  for (auto I = idx; I < N; I += stride) {
    out[I] = I + data.x + data.y + data.z;
  }
}

int main(int argc, const char *argv[]) {
  int elements = 4000;
  A data{1, 1, 1};
  float *out;
  float *h_out = new float[elements];

  auto EC = MnemeDeviceRT::DeviceErrorCheck(
      MnemeDeviceRT::DeviceMalloc((void **)&out, sizeof(float) * elements));
  if (EC) {
    std::cout << "Error when mallocing in test benchmark " << EC.value()
              << "\n";
    return -1;
  }

  dim3 blockDim(256, 1, 1);
  dim3 gridDim((elements + 255) / 256, 1, 1);
  test<<<gridDim, blockDim>>>(data, out, elements);
  EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceSynchronize());
  if (EC) {
    std::cout << "Error when running kernel" << EC.value() << "\n";
    return -1;
  }

  EC = MnemeDeviceRT::DeviceErrorCheck(
      MnemeDeviceRT::DeviceCopy((void *)h_out, out, sizeof(float) * elements,
                                MnemeDeviceRT::MemcpyDeviceToHostKind()));
  if (EC) {
    std::cout << "Could not copy from device " + EC.value() << "\n";
    return -1;
  }

  EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceFree((void *)out));
  if (EC) {
    std::cout << "Could not copy from device " + EC.value() << "\n";
    return -1;
  }

  for (int I = 0; I < elements; I++) {
    auto tmp = I + data.x + data.y + data.z;
    if (tmp != h_out[I]) {
      std::cout << "Values differ " << tmp << " gpu " << h_out[I] << "\n";
      return -1;
    }
  }

  delete[] h_out;
  std::cout << "Correct data\n";

  return 0;
}

// CHECK: Correct data
