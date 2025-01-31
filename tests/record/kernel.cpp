// clang-format off
// RUN: LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_LOG_LEVEL=debug MNEME_PAGE_SIZE=%PG ./kernel | FileCheck %s --check-prefixes=CHECK
// clang-format on

#include <climits>
#include <cstdio>

#include "DeviceTraits.hpp"
using namespace mneme;

#ifdef ENABLE_HIP
using MnemeDeviceRT = DeviceTraits<DeviceVendors::HIP>;
#endif

__global__ void kernel() { printf("Kernel\n"); }

int main() {
  kernel<<<1, 1>>>();
  auto EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceSynchronize());
  if (EC) {
    std::cout << "Error when running benchmark " << EC.value() << "\n";
    return -1;
  }
  return 0;
}

// CHECK: Kernel
