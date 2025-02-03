#include <stdio.h>

#include "DeviceTraits.hpp"
using namespace mneme;

#ifdef ENABLE_HIP
using MnemeDeviceRT = DeviceTraits<DeviceVendors::HIP>;
#endif

__global__ static void kernel() { printf("File2-Kernel\n"); }

int foo() {
  kernel<<<1, 1>>>();
  auto EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceSynchronize());
  if (EC) {
    std::cout << "Error when calling kernel from foo" << EC.value() << "\n";
    return -1;
  }
  return 0;
}
