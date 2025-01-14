#pragma once
#include "Utils.hpp"
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
