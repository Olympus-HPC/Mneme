#include "mneme/DeviceTraits.hpp"
#include <cstring>
#include <llvm/ADT/DenseMap.h>

using namespace mneme;

#ifdef MNEME_ENABLE_HIP
constexpr DeviceVendors Vendor = DeviceVendors::HIP;
using MnemeDeviceRT = DeviceTraits<DeviceVendors::HIP>;
#elif defined(MNEME_ENABLE_CUDA)
constexpr DeviceVendors Vendor = DeviceVendors::CUDA;
using MnemeDeviceRT = DeviceTraits<DeviceVendors::CUDA>;
#endif

int main(int argc, char *argv[]) {
  {
    char *DevPtr[2];
    int NumBytes = 544;
    for (int i = 0; i < 2; i++) {
      auto EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceMalloc(
          reinterpret_cast<void **>(&DevPtr[i]), NumBytes));
      if (EC)
        LOG_FATAL("Error in comparing blobs " + EC.value());

      EC = MnemeDeviceRT::DeviceErrorCheck(
          MnemeDeviceRT::DeviceMemset(DevPtr[i], 3, NumBytes));

      if (EC)
        LOG_FATAL("Error in comparing blobs " + EC.value());
    }
    if (!MnemeDeviceRT::compareDeviceBlobs(DevPtr[0], DevPtr[1], NumBytes)) {
      std::cout << "Expected memory to be same";
      return -1;
    }

    for (int i = 0; i < 2; i++) {
      auto EC =
          MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceFree(DevPtr[i]));
      if (EC)
        LOG_FATAL("Error in comparing blobs " + EC.value());
    }
  }

  {
    char *DevPtr[2];
    int NumBytes = 544;
    for (int i = 0; i < 2; i++) {
      auto EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceMalloc(
          reinterpret_cast<void **>(&DevPtr[i]), NumBytes));
      if (EC)
        LOG_FATAL("Error in comparing blobs " + EC.value());

      EC = MnemeDeviceRT::DeviceErrorCheck(
          MnemeDeviceRT::DeviceMemset(DevPtr[i], 3 + i, NumBytes));

      if (EC)
        LOG_FATAL("Error in comparing blobs " + EC.value());
    }
    if (MnemeDeviceRT::compareDeviceBlobs(DevPtr[0], DevPtr[1], NumBytes)) {
      std::cout << "Expected memory to differ";
      return -1;
    }

    for (int i = 0; i < 2; i++) {
      auto EC =
          MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceFree(DevPtr[i]));
      if (EC)
        LOG_FATAL("Error in comparing blobs " + EC.value());
    }
  }
  return 0;
}
