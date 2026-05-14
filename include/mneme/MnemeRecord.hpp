#pragma once

#include "mneme/MnemeConfig.hpp"
#include "mneme/MnemeNullRecorder.hpp"
#include "mneme/MnemeRecorderBackend.hpp"
#include "mneme/MnemeRecordingBackend.hpp"

#include <memory>
#include <utility>

namespace mneme {

template <DeviceVendors VendorTypes> class MnemeRecorder {
public:
  using MnemeDeviceRT = DeviceTraits<VendorTypes>;
  using DeviceError_t = typename MnemeDeviceRT::DeviceError_t;
  using DeviceStream_t = typename MnemeDeviceRT::DeviceStream_t;
  using KernelFunction_t = typename MnemeDeviceRT::KernelFunction_t;

private:
  std::unique_ptr<RecorderBackend<VendorTypes>> Backend;

public:
  MnemeRecorder() {
    if (Config::get().isRecordingEnabledForCurrentRank())
      Backend = std::make_unique<RecordingBackend<VendorTypes>>();
    else
      Backend = std::make_unique<NullRecorder<VendorTypes>>();
  }

  bool setMetadataForPointer(const void *ptr, Metadata md) {
    if (!Backend) return false;
    return Backend->setMetadataForPointer(ptr, std::move(md));
  }

  bool getMetadataForPointer(const void *ptr, Metadata &md) const {
    if (!Backend) return false;
    return Backend->getMetadataForPointer(ptr, md);
  }

  bool eraseMetadataForPointer(const void *ptr) {
    if (!Backend) return false;
    return Backend->eraseMetadataForPointer(ptr);
  }

  DeviceError_t rtMalloc(void **ptr, size_t size) {
    if (!Backend) return MnemeDeviceRT::DeviceSuccess;
    return Backend->rtMalloc(ptr, size);
  }

  DeviceError_t rtManagedMalloc(void **ptr, size_t size, unsigned int flags) {
    if (!Backend) return MnemeDeviceRT::DeviceSuccess;
    return Backend->rtManagedMalloc(ptr, size, flags);
  }

  DeviceError_t rtHostMalloc(void **ptr, size_t size, unsigned int flags) {
    if (!Backend) return MnemeDeviceRT::DeviceSuccess;
    return Backend->rtHostMalloc(ptr, size, flags);
  }

  DeviceError_t rtFree(void *ptr) {
    if (!Backend) return MnemeDeviceRT::DeviceSuccess;
    return Backend->rtFree(ptr);
  }

  DeviceError_t rtHostFree(void *ptr) {
    if (!Backend) return MnemeDeviceRT::DeviceSuccess;
    return Backend->rtHostFree(ptr);
  }

  DeviceError_t rtLaunchKernel(const void *func, dim3 &GridDim, dim3 &BlockDim,
                               void **Args, size_t SharedMem,
                               DeviceStream_t Stream) {
    if (!Backend) return MnemeDeviceRT::DeviceSuccess;
    return Backend->rtLaunchKernel(func, GridDim, BlockDim, Args, SharedMem,
                                   Stream);
  }

  DeviceError_t rtSetDevice(int deviceID) {
    if (!Backend) {
      // Backend destroyed during shutdown - return success
      return MnemeDeviceRT::DeviceSuccess;
    }
    return Backend->rtSetDevice(deviceID);
  }

  DeviceError_t rtGetDevice(int *deviceID) {
    if (!Backend) {
      // Backend destroyed during shutdown - set device to 0 and return success
      if (deviceID) *deviceID = 0;
      return MnemeDeviceRT::DeviceSuccess;
    }
    return Backend->rtGetDevice(deviceID);
  }
};

} // namespace mneme
