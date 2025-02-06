#include "mneme/DeviceTraits.hpp"
#include <cstring>
#include <llvm/ADT/DenseMap.h>

#ifdef ENABLE_HIP
#include "mneme/MnemeMemoryHIP.hpp"
#include "mneme/MnemeRecordHIP.hpp"
#endif

using namespace mneme;

using MnemeRecorderDevice = MnemeRecorderHIP;
using MnemeMemoryBlobDevice = MnemeMemoryBlob<DeviceVendors::HIP>;
using DeviceVendorTraits = DeviceTraits<DeviceVendors::HIP>;

int main(int argc, char *argv[]) {
  {
    char *DevPtr[2];
    int NumBytes = 544;
    for (int i = 0; i < 2; i++) {
      auto EC =
          DeviceTraits<HIP>::DeviceErrorCheck(DeviceTraits<HIP>::DeviceMalloc(
              reinterpret_cast<void **>(&DevPtr[i]), NumBytes));
      if (EC)
        FATAL_ERROR("Error in comparing blobs " + EC.value());

      EC = DeviceTraits<HIP>::DeviceErrorCheck(
          DeviceTraits<HIP>::DeviceMemset(DevPtr[i], 3, NumBytes));

      if (EC)
        FATAL_ERROR("Error in comparing blobs " + EC.value());
    }
    if (!DeviceTraits<HIP>::compareDeviceBlobs(DevPtr[0], DevPtr[1],
                                               NumBytes)) {
      std::cout << "Expected memory to be same";
      return -1;
    }

    for (int i = 0; i < 2; i++) {
      auto EC = DeviceTraits<HIP>::DeviceErrorCheck(
          DeviceTraits<HIP>::DeviceFree(DevPtr[i]));
      if (EC)
        FATAL_ERROR("Error in comparing blobs " + EC.value());
    }
  }

  {
    char *DevPtr[2];
    int NumBytes = 544;
    for (int i = 0; i < 2; i++) {
      auto EC =
          DeviceTraits<HIP>::DeviceErrorCheck(DeviceTraits<HIP>::DeviceMalloc(
              reinterpret_cast<void **>(&DevPtr[i]), NumBytes));
      if (EC)
        FATAL_ERROR("Error in comparing blobs " + EC.value());

      EC = DeviceTraits<HIP>::DeviceErrorCheck(
          DeviceTraits<HIP>::DeviceMemset(DevPtr[i], 3 + i, NumBytes));

      if (EC)
        FATAL_ERROR("Error in comparing blobs " + EC.value());
    }
    if (DeviceTraits<HIP>::compareDeviceBlobs(DevPtr[0], DevPtr[1], NumBytes)) {
      std::cout << "Expected memory to differ";
      return -1;
    }

    for (int i = 0; i < 2; i++) {
      auto EC = DeviceTraits<HIP>::DeviceErrorCheck(
          DeviceTraits<HIP>::DeviceFree(DevPtr[i]));
      if (EC)
        FATAL_ERROR("Error in comparing blobs " + EC.value());
    }
  }
  return 0;
}
