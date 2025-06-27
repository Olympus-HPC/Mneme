// clang-format off
// RUN: rm -rf *.json Recorded*.bc DeviceState*.mneme
// RUN: LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_LOG_LEVEL=off MNEME_PAGE_SIZE=%PG ./test_multi_file%ext | FileCheck %s --check-prefixes=CHECK
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

// CHECK-RR:DemangledName: kernel()
// CHECK-RR:NumModules: 1
// CHECK-RR:NumInstances: 1
// CHECK-RR:DemangledName: kernel()
// CHECK-RR:NumModules: 1
// CHECK-RR:NumInstances: 1
__global__ static void kernel() { printf("File1-Kernel\n"); }

int foo();
int main() {
  kernel<<<1, 1>>>();
  auto EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceSynchronize());
  if (EC) {
    std::cout << "Error when calling kernel" << EC.value() << "\n";
    return -1;
  }
  return foo();
}

// CHECK:File1-Kernel
// CHECK:File2-Kernel
