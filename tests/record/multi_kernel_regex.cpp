// clang-format off
// RUN: rm -rf *.json Recorded*.bc DeviceState*.mneme
// RUN: LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_LOG_LEVEL=debug MNEME_PAGE_SIZE=%PG ./multi_kernel_regex%ext | FileCheck %s --check-prefixes=CHECK
// RUN: %RR | FileCheck %s --check-prefix=CHECK-RR-NOREGEX
// RUN: rm -rf *.json Recorded*.bc DeviceState*.mneme
// RUN: MNEME_RR_KERNELS="_two" LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_LOG_LEVEL=debug MNEME_PAGE_SIZE=%PG ./multi_kernel_regex%ext | FileCheck %s --check-prefixes=CHECK
// RUN: %RR | FileCheck %s --check-prefix=CHECK-RR-WITH-REGEX
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

// CHECK-RR-NOREGEX-DAG: DemangledName: kernel_one()
// CHECK-RR-NOREGEX-DAG: NumModules: 1
// CHECK-RR-NOREGEX-DAG: NumInstances: 1
// CHECK-RR-NOREGEX-DAG: DemangledName: kernel_two()
// CHECK-RR-NOREGEX-DAG: NumModules: 1
// CHECK-RR-NOREGEX-DAG: NumInstances: 1

// CHECK-RR-WITH-REGEX-NOT: DemangledName: kernel_one()
// CHECK-RR-WITH-REGEX-DAG: DemangledName: kernel_two()
// CHECK-RR-WITH-REGEX-DAG: NumModules: 1
// CHECK-RR-WITH-REGEX-DAG: NumInstances: 1

__global__ void kernel_one() { printf("Kernel One\n"); }
__global__ void kernel_two() { printf("Kernel Two\n"); }

int main() {
  // CHECK-RR-NOREGEX-DAG: BlockDims:(1, 1, 1)
  // CHECK-RR-NOREGEX-DAG: GridDims:(1, 1, 1)
  kernel_one<<<1, 1>>>();
  auto EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceSynchronize());
  if (EC) {
    std::cout << "Error when running benchmark " << EC.value() << "\n";
    return -1;
  }
  // CHECK-RR-NOREGEX-DAG: BlockDims:(1, 1, 1)
  // CHECK-RR-NOREGEX-DAG: GridDims:(1, 1, 1)
  kernel_two<<<1, 1>>>();
  EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceSynchronize());
  if (EC) {
    std::cout << "Error when running benchmark " << EC.value() << "\n";
    return -1;
  }

  return 0;
}

// CHECK: Kernel One
// CHECK: Kernel Two
