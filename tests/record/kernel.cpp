// clang-format off
// RUN: rm -rf *.json Recorded*.bc DeviceState*.mneme
// RUN: LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_LOG_LEVEL=debug MNEME_PAGE_SIZE=%PG ./kernel%ext | FileCheck %s --check-prefixes=CHECK
// RUN: %RR | FileCheck %s --check-prefix=CHECK-RR
// RUN: rm -rf *.json Recorded*.bc DeviceState*.mneme
// clang-format on

#include <climits>
#include <cstdio>

#include "mneme/DeviceTraits.hpp"
using namespace mneme;

#ifdef MNEME_ENABLE_HIP
using MnemeDeviceRT = DeviceTraits<DeviceVendors::HIP>;
#elif defined(MNEME_ENABLE_CUDA)
using MnemeDeviceRT = DeviceTraits<DeviceVendors::CUDA>;
#endif

__device__ int var = 0;

// CHECK-RR:DemangledName: kernel()
// CHECK-RR:NumModules: 1
// CHECK-RR:NumInstances: 1
__global__ void kernel(int a) { printf("Kernel %d %d\n", var, a); }

int main() {
  // CHECK-RR:BlockDims:(1, 1, 1)
  // CHECK-RR:GridDims:(1, 1, 1)
  kernel<<<1, 1>>>(5);
  auto EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceSynchronize());
  if (EC) {
    std::cout << "Error when running benchmark " << EC.value() << "\n";
    return -1;
  }
  return 0;
}

// CHECK: Kernel
