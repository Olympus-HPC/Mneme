#pragma once
#include "Utils.hpp"
#include <hip/hip_runtime.h>
namespace mneme {
enum DeviceVendors { HIP, CUDA };

template <DeviceVendors Type> struct DeviceTraits;

#ifdef ENABLE_HIP
template <> struct DeviceTraits<DeviceVendors::HIP> {
  using DeviceError_t = hipError_t;
  using DeviceStream_t = hipStream_t;
  using KernelFunction_t = hipFunction_t;
  using MemoryAllocationHandle_t = hipMemGenericAllocationHandle_t;

  static inline std::optional<std::string>
  DeviceErrorCheck(hipError_t ErrorCode) {
    if (ErrorCode == hipSuccess)
      return std::nullopt;
    return std::string(hipGetErrorString(ErrorCode));
  }

  static hipError_t DeviceStreamSynchronize(hipStream_t Stream) {
    return hipStreamSynchronize(Stream);
  }

  static hipError_t DeviceMalloc(void **ptr, size_t size) {
    return hipMalloc(ptr, size);
  }

  static hipError_t DeviceMemcpy();

  static hipError_t DeviceFree(void *ptr) { return hipFree(ptr); }

  static constexpr hipMemcpyKind MemcpyHostToDeviceKind() {
    return hipMemcpyHostToDevice;
  }

  static hipError_t DeviceCopy(void *Dest, void *Src, size_t SizeBytes,
                               hipMemcpyKind Kind) {
    return hipMemcpy(Dest, Src, SizeBytes, Kind);
  }
};
#elif defined(ENABLE_CUDA)
template <> struct DeviceTraits<DeviceVendors::HIP> {
  using DeviceError_t = hipError_t;
  using DeviceStream_t = hipStream_t;
  using KernelFunction_t = hipFunction_t;
  using AllocGranularityFlags = hipMemAllocationGranularity_flags;
};
#else
#endif

} // namespace mneme
