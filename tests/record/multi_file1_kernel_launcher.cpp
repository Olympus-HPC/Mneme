// clang-format off
// clang-format off
// RUN: rm -rf *.json Recorded*.bc DeviceState*.mneme
// RUN: LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_LOG_LEVEL=off MNEME_PAGE_SIZE=%PG ./test_multi_file_launcher%ext | FileCheck %s --check-prefixes=CHECK
// RUN: %RR | FileCheck %s --check-prefix=CHECK-RR
// RUN: rm -rf *.json Recorded*.bc DeviceState*.mneme
// clang-format on#include <climits>
//
#include <cstdio>

#include "mneme/DeviceTraits.hpp"
#include "launcher.hpp"
using namespace mneme;

#ifdef MNEME_ENABLE_HIP
using MnemeDeviceRT = DeviceTraits<DeviceVendors::HIP>;
#endif

// CHECK-RR:DemangledName: void kernel<kernel_body_t>(kernel_body_t) 
// CHECK-RR:NumModules: 1
// CHECK-RR:NumInstances: 1

int foo();
int main() {
  auto EC = MnemeDeviceRT::DeviceErrorCheck(launcher(kernel_body));
  if (EC) {
    std::cout << "Error when calling kernel" << EC.value() << "\n";
    return -1;
  }

  EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceSynchronize());
  if (EC) {
    std::cout << "Error when calling kernel" << EC.value() << "\n";
    return -1;
  }

  return foo();
}

// CHECK: Kernel body
// CHECK: Kernel body
