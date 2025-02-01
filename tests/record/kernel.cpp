// clang-format off
// RUN: rm -rf *.json Recorded*.bv DEviceState*.mneme
// RUN: LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_LOG_LEVEL=debug MNEME_PAGE_SIZE=%PG ./kernel | FileCheck %s --check-prefixes=CHECK
// RUN: %RR | FileCheck %s --check-prefix=CHECK-RR
// RUN: rm -rf *.json Recorded*.bv DEviceState*.mneme
// clang-format on

#include <climits>
#include <cstdio>

#include "DeviceTraits.hpp"
using namespace mneme;

#ifdef ENABLE_HIP
using MnemeDeviceRT = DeviceTraits<DeviceVendors::HIP>;
#endif

// CHECK-RR:DemangledName: kernel()
// CHECK-RR:NumModules: 1
// CHECK-RR:NumInstances: 1
__global__ void kernel() { printf("Kernel\n"); }

int main() {
  // CHECK-RR:BlockDims:(1, 1, 1)
  // CHECK-RR:GridDims:(1, 1, 1)
  kernel<<<1, 1>>>();
  auto EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceSynchronize());
  if (EC) {
    std::cout << "Error when running benchmark " << EC.value() << "\n";
    return -1;
  }
  return 0;
}

// CHECK: Kernel
