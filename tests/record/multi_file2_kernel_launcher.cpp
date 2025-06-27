#include <stdio.h>

#include "launcher.hpp"
#include "mneme/DeviceTraits.hpp"
using namespace mneme;

#ifdef MNEME_ENABLE_HIP
using MnemeDeviceRT = DeviceTraits<DeviceVendors::HIP>;
#elif defined(MNEME_ENABLE_CUDA)
using MnemeDeviceRT = DeviceTraits<DeviceVendors::CUDA>;
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
