#pragma once
#include "Utils.hpp"
#include <hip/hip_runtime.h>
namespace mneme {
enum DeviceVendors { HIP, CUDA };

template <DeviceVendors Type> struct DeviceTraits;

#if defined(ENABLE_HIP)
template <> struct DeviceTraits<DeviceVendors::HIP> {
  using DeviceError_t = hipError_t;
  using DeviceStream_t = hipStream_t;
  using KernelFunction_t = hipFunction_t;
  using MemoryAllocationHandle_t = hipMemGenericAllocationHandle_t;
  using DeviceModule_t = hipModule_t;
  using DevicePtr_t = hipDeviceptr_t;
  using DeviceHandle_t = hipDevice_t;
  using DeviceContext_t = hipCtx_t;
  using DeviceFunction_t = hipFunction_t;
  using DeviceEvent_t = hipEvent_t;
  static constexpr auto DeviceSuccess = hipSuccess;

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

  static hipError_t DeviceFree(void *ptr) { return hipFree(ptr); }

  static constexpr hipMemcpyKind MemcpyHostToDeviceKind() {
    return hipMemcpyHostToDevice;
  }

  static hipError_t DeviceCopy(void *Dest, void *Src, size_t SizeBytes,
                               hipMemcpyKind Kind) {
    return hipMemcpy(Dest, Src, SizeBytes, Kind);
  }

  static std::string GetDeviceArch() {
    DeviceHandle_t Dev;
    DeviceContext_t Ctx;
    auto EC = DeviceErrorCheck(hipInit(0));
    if (EC)
      FATAL_ERROR("Could not initialize device\n EC:" + EC.value());

    EC = DeviceErrorCheck(hipGetDevice(&Dev));
    if (EC)
      FATAL_ERROR("Could not get device\n EC:" + EC.value());

    hipDeviceProp_t device_prop;

    // Get properties of the current device
    EC = DeviceErrorCheck(hipGetDeviceProperties(&device_prop, Dev));
    if (EC)
      FATAL_ERROR("Could not get device properties\n EC:" + EC.value());

    std::string arch_name = device_prop.gcnArchName;

    std::string HipArch = arch_name.substr(0, arch_name.find(':'));

    return std::string(HipArch);
  }

  static hipModule_t getDeviceModuleFromImage(const void *Image) {
    hipModule_t HipModule;

    auto EC = DeviceErrorCheck(hipModuleLoadData(&HipModule, Image));
    if (EC)
      FATAL_ERROR("Error with loading data from module\nEC:" + EC.value());
    return HipModule;
  }

  static std::pair<void *, size_t>
  getGlobalAddrFromModule(hipModule_t &HipModule, std::string &GlobalName) {
    size_t Size;
    hipDeviceptr_t DevPtr;
    auto EC = DeviceErrorCheck(
        hipModuleGetGlobal(&DevPtr, &Size, HipModule, GlobalName.c_str()));
    if (EC)
      FATAL_ERROR("Could not load global variable '" + GlobalName +
                  "' from device module\n:EC:" + EC.value());
    return std::make_pair((void *)DevPtr, Size);
  }

  static hipFunction_t getKernelFunctionFromImage(hipModule_t &HipModule,
                                                  std::string &KernelName) {
    hipFunction_t KernelFunc;
    auto EC = DeviceErrorCheck(
        hipModuleGetFunction(&KernelFunc, HipModule, KernelName.c_str()));
    if (EC)
      FATAL_ERROR("Error with loading kernel from Module");

    return KernelFunc;
  }

  static hipError_t launchKernelFunction(hipFunction_t KernelFunc, dim3 GridDim,
                                         dim3 BlockDim, void **KernelArgs,
                                         uint64_t ShmemSize,
                                         hipStream_t Stream) {
    return hipModuleLaunchKernel(KernelFunc, GridDim.x, GridDim.y, GridDim.z,
                                 BlockDim.x, BlockDim.y, BlockDim.z, ShmemSize,
                                 Stream, KernelArgs, nullptr);
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
