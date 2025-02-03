#include <stdio.h>

#include "DeviceTraits.hpp"
#include "launcher.hpp"
using namespace mneme;

#ifdef ENABLE_HIP
using MnemeDeviceRT = DeviceTraits<DeviceVendors::HIP>;
#endif

int foo() {
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

  return 0;
}
