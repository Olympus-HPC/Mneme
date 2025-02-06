#pragma once
#include "mneme/MnemeLogger.hpp"
#include "mneme/Utils.hpp"
#include <hip/hip_runtime.h>
#include <optional>

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

  static hipError_t DeviceMemset(void *DevPtr, int Value, size_t Bytes) {
    auto EC = hipMemset(DevPtr, Value, Bytes);
    return EC;
  }

  static hipError_t DeviceMalloc(void **ptr, size_t size) {
    return hipMalloc(ptr, size);
  }

  static hipError_t DeviceFree(void *ptr) { return hipFree(ptr); }

  static hipError_t DeviceCopy(void *Dest, void *Src, size_t SizeBytes,
                               hipMemcpyKind Kind) {
    return hipMemcpy(Dest, Src, SizeBytes, Kind);
  }

  static hipError_t DeviceSynchronize() { return hipDeviceSynchronize(); }

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

  static inline uint64_t
  getPageSize(int DeviceID,
              const hipMemAllocationGranularity_flags Granularity) {
    uint64_t PageSize;
    hipMemAllocationProp Prop = {};
    Prop.type = hipMemAllocationTypePinned;
    Prop.location.type = hipMemLocationTypeDevice;
    Prop.location.id = DeviceID;
    // TODO: I could not find any documentation regarding the compressionType in
    // HIP. I will leave unitialized a.t.m.
    // Prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;

    hipErrCheck(hipMemGetAllocationGranularity(&PageSize, &Prop, Granularity));
    return PageSize;
  }

  static constexpr hipMemcpyKind MemcpyHostToDeviceKind() {
    return hipMemcpyHostToDevice;
  }

  static constexpr hipMemcpyKind MemcpyDeviceToHostKind() {
    return hipMemcpyDeviceToHost;
  }

  static void mmap(hipMemGenericAllocationHandle_t &MHandle, void *Addr,
                   uintptr_t Size, int DeviceID) {
    hipMemAllocationProp Prop = {};
    Prop.type = hipMemAllocationTypePinned;
    Prop.location.type = hipMemLocationTypeDevice;
    Prop.location.id = DeviceID;
    hipErrCheck(hipMemCreate(&MHandle, Size, &Prop, 0));
    hipErrCheck(hipMemMap((void *)Addr, Size, 0, MHandle, 0));

    hipMemAccessDesc ADesc = {};
    ADesc.location.type = hipMemLocationTypeDevice;
    ADesc.location.id = DeviceID;
    ADesc.flags = hipMemAccessFlagsProtReadWrite;

    hipErrCheck(hipMemSetAccess(Addr, Size, &ADesc, 1));
  }

  static uint64_t getMinPageSize(int DeviceID) {
    return getPageSize(DeviceID, hipMemAllocationGranularityMinimum);
  }

  static void *getVirtualAddress(uint64_t Size, void *VA, uint64_t Alignment) {
    hipDeviceptr_t devPtr = 0;

    hipErrCheck(hipMemAddressReserve(&devPtr, Size, Alignment,
                                     reinterpret_cast<hipDeviceptr_t>(VA), 0));
    return (void *)devPtr;
  }

  static void unmap(hipMemGenericAllocationHandle_t &MHandle, void *Addr,
                    uintptr_t Size) {
    LOG_DEBUG("Unmapping Addr:{} SIZE:{}", Addr, Size);
    hipErrCheck(hipMemUnmap(Addr, Size));
    hipErrCheck(hipMemRelease(MHandle));
  }

  static size_t getFixedMemorySize() {
    static uint64_t PageSize{[&]() {
      const char *env_p = std::getenv("MNEME_PAGE_SIZE");
      if (!env_p)
        return static_cast<uint64_t>(64L * 1024L * 1024L * 1024L);
      return static_cast<uint64_t>(std::atol(env_p) * 1024L * 1024L * 1024L);
    }()};
    return PageSize;
  }

  static void freeVirtualAddress(void *Addr, size_t Size) {
    LOG_DEBUG("Releasing Device Virtual Address Pages:{} Size:{}", Addr, Size);
    auto EC = DeviceErrorCheck(hipMemAddressFree(Addr, Size));
    if (EC) {
      FATAL_ERROR("Could not release VA addresses " + EC.value());
    }
  }

  static bool compareDeviceBlobs(const char *Blob1, const char *Blob2,
                                 uint64_t NumBytes);

  static hipError_t DeviceStreamCreate(hipStream_t *Stream) {
    return hipStreamCreate(Stream);
  }

  static constexpr uintptr_t getSuggestedAddr() { return 0x0000153d2be00000; }

  static hipError_t deviceLaunchKernel(const void *function_address,
                                       dim3 numBlocks, dim3 dimBlocks,
                                       void **args, size_t sharedMemBytes,
                                       hipStream_t stream) {
    return hipLaunchKernel(function_address, numBlocks, dimBlocks, args,
                           sharedMemBytes, stream);
  }

  static hipError_t deviceGetSymbolAddress(void **devPtr, const void *symbol) {
    return hipGetSymbolAddress(devPtr, symbol);
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
