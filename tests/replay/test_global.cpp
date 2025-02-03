// clang-format off
// RUN: rm -rf *.json Recorded*.bc DeviceState*.mneme
// RUN: LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_LOG_LEVEL=debug MNEME_PAGE_SIZE=%PG ./kernel%ext | FileCheck %s --check-prefixes=CHECK
// RUN: %RR | FileCheck %s --check-prefix=CHECK-RR
// RUN: rm -rf *.json Recorded*.bc DeviceState*.mneme
// clang-format on

#include <climits>
#include <cstdio>

#include "DeviceTraits.hpp"
#include "Utils.hpp"
using namespace mneme;

#ifdef ENABLE_HIP
using MnemeDeviceRT = DeviceTraits<DeviceVendors::HIP>;
#define SYMBOL(x) HIP_SYMBOL(x)
#endif

__device__ double value = 0.0;

__global__ void test_global() { value = 1.0; }

int main(int argc, const char *argv[]) {
  test_global<<<1, 1>>>();
  auto EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceSynchronize());
  if (EC) {
    std::cout << "Error on launching kernel " << EC.value() << "\n";
    return -1;
  }

  double *devPtr;
  EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::deviceGetSymbolAddress(
      (void **)&devPtr, (const char *)&value));
  if (EC) {
    std::cout << "Could not load symbol address\n";
    return -1;
  }
  double hValue;
  EC = MnemeDeviceRT::DeviceErrorCheck(
      MnemeDeviceRT::DeviceCopy((void *)&hValue, devPtr, sizeof(double),
                                MnemeDeviceRT::MemcpyDeviceToHostKind()));
  if (EC) {
    std::cout << "Could not copy from device " + EC.value() << "\n";
    return -1;
  }

  if (hValue != 1.0) {
    std::cout << "Expected the value to be 1.0 but is " << hValue << "\n";
    return -1;
  }
  std::cout << "Value is " << hValue << "\n";
  return 0;
}
