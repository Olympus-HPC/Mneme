#pragma once

#include "mneme/MnemeLLVMUtils.hpp"
#include "mneme/MnemeMemory.hpp"
#include "mneme/MnemePageManager.hpp"
#include "mneme/MnemeUtils.hpp"
#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <dlfcn.h>

#include "llvm/Support/raw_ostream.h"
#include <filesystem>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StableHashing.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <mutex>

#include <proteus/CompilerInterfaceDevice.h>
#include <proteus/JitEngineDevice.hpp>

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeKernelInfo.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeSnapshot.hpp"
#include "mneme/MnemeSymbols.hpp"

namespace mneme {

template <DeviceVendors VendorTypes> class MnemeRecorder {
protected:
  void *rtLib;
  void *proteusLib;
  std::string RecordReplayDir;
  llvm::DenseMap<void **, llvm::SmallVector<std::shared_ptr<KernelInfo>>>
      HandleToKernels;
  llvm::DenseMap<const void *, std::shared_ptr<KernelInfo>> KernelInfoMap;
  llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> AllocatedBlobs;

  std::unique_ptr<PageManager<VendorTypes>> PM;
  void *VAStartAddr;
  int64_t VATotalSize;

  // NOTE: We only keep track of the first time we set the device id. Once we
  // create the allocator we assume that the allocations go to the same device
  int DeviceID;

public:
  using MnemeDeviceRT = DeviceTraits<VendorTypes>;
  using DeviceError_t = typename MnemeDeviceRT::DeviceError_t;
  using DeviceStream_t = typename MnemeDeviceRT::DeviceStream_t;
  using KernelFunction_t = typename MnemeDeviceRT::KernelFunction_t;

private:
  bool ExtractedIR;
  RecordDatabase DB;
  std::once_flag ExtractFlag;

  DeviceError_t (*origLaunchKernel)(const void *func, dim3 gridDim,
                                    dim3 blockDim, void **args,
                                    size_t sharedMem,
                                    DeviceStream_t stream) = nullptr;

  DeviceError_t (*proteusLaunchKernel)(const void *func, dim3 gridDim,
                                       dim3 blockDim, void **args,
                                       size_t sharedMem,
                                       DeviceStream_t stream) = nullptr;

  DeviceError_t (*origMallocDevice)(void **ptr, size_t size);

  DeviceError_t (*origMallocPinned)(void **ptr, size_t size,
                                    unsigned int flags);

  DeviceError_t (*origMallocManaged)(void **ptr, size_t size,
                                     unsigned int flags);

  DeviceError_t (*origFreeDevice)(void *devPtr);

  DeviceError_t (*origFreeHost)(void *ptr);

  DeviceError_t (*origSetDeviceID)(int id);
  DeviceError_t (*origGetDeviceID)(int *id);

public:
  DeviceError_t rtMalloc(void **ptr, size_t size) {
    if (!PM) {
      // NOTE: We need this arch cause internally we initialize the device.
      // FIXME: We need to have a DeviceTrait function to initialize the GPU
      // and call it separately here. Let's do this on a separate PR
      auto arch = MnemeDeviceRT::GetDeviceArch();
      LOG_DEBUG("Initializing system {}", arch);
      if (DeviceID == -1) {
        origGetDeviceID(&DeviceID);
      }
      PM = initializePageManager<VendorTypes>(
          DeviceID, (void *)MnemeDeviceRT::getSuggestedAddr());
    }

    auto [Addr, ReservedSize] = PM->allocateAddr(size, nullptr);
    MnemeMemoryBlob<VendorTypes> MemBlob(ReservedSize,
                                         reinterpret_cast<void *>(Addr), size);
    auto ret = MemBlob.map(reinterpret_cast<void *>(Addr), ReservedSize, size);
    *ptr = MemBlob.ptr();
    AllocatedBlobs.insert({*ptr, std::move(MemBlob)});
    LOG_DEBUG("Intercepted Device Malloc PTR:{} SIZE:{} ACTUALSIZE:{}", *ptr,
              size, ReservedSize);
    return ret;
  };

  DeviceError_t rtManagedMalloc(void **ptr, size_t size, unsigned int flags) {
    auto ret = origMallocManaged(ptr, size, flags);
    LOG_DEBUG("Intercepted Managed Malloc PTR:{} SIZE:{}", *ptr, size);
    LOG_WARN("Will not be able to replay Kernels acessing:{}", *ptr);
    return ret;
  };

  DeviceError_t rtHostMalloc(void **ptr, size_t size, unsigned int flags) {
    auto ret = origMallocPinned(ptr, size, flags);
    LOG_WARN("Intercepted Pinned|Host Malloc PTR:{} SIZE:{}", *ptr, size);
    return ret;
  }

  DeviceError_t rtFree(void *ptr) {
    if (ptr == nullptr) {
      LOG_WARN("Mneme was instructed to de-allocate nullptr..., skipping");
      return MnemeDeviceRT::DeviceSuccess;
    }
    if (!AllocatedBlobs.contains(ptr)) {
      LOG_CRITICAL("Free address that is not being allocated through Mneme {}",
                   ptr);
      LOG_FATAL("Free address that is not being allocated through Mneme\n");
    }
    PM->releaseAddr(AllocatedBlobs[ptr].getActualSize(), ptr);
    auto ret = AllocatedBlobs[ptr].release();
    LOG_DEBUG("Intercepted device Free PTR:{} SIZE:{} ACTUALSIZE:{}", ptr,
              AllocatedBlobs[ptr].getSize(),
              AllocatedBlobs[ptr].getActualSize());
    AllocatedBlobs.erase(ptr);
    return ret;
  };

  DeviceError_t rtHostFree(void *ptr) {
    auto ret = origFreeHost(ptr);
    LOG_DEBUG("Free pinned address:{}", ptr);
    return ret;
  }

  DeviceError_t rtLaunchKernel(const void *func, dim3 &GridDim, dim3 &BlockDim,
                               void **Args, size_t SharedMem,
                               DeviceStream_t Stream) {
    using namespace llvm;
    using namespace proteus;

    if (!PM) {
      // NOTE: We need this arch cause internally we initialize the device.
      // FIXME: We need to have a DeviceTrait function to initialize the GPU
      // and call it separately here. Let's do this on a separate PR
      auto arch = MnemeDeviceRT::GetDeviceArch();
      if (DeviceID == -1) {
        origGetDeviceID(&DeviceID);
      }
      LOG_DEBUG("Initializing system {}", arch);
      PM = initializePageManager<VendorTypes>(
          DeviceID, (void *)MnemeDeviceRT::getSuggestedAddr());
    }

    auto &Proteus = JitDeviceImplT::instance();
    auto OptionalKernelInfo = Proteus.getJITKernelInfo(func);
    LOG_DEBUG("Received OptionalKernel Info {}", (void *)origLaunchKernel);
    if (!OptionalKernelInfo) {
      LOG_DEBUG("Information for kernel  {} is not included", func);
      return origLaunchKernel(func, GridDim, BlockDim, Args, SharedMem, Stream);
    }
    auto &KInfo = OptionalKernelInfo.value().get();
    auto &BinInfo = KInfo.getBinaryInfo();
    BinInfo.mapGlobals();
    LOG_DEBUG("Continue with {}", KInfo.getName());
    Proteus.extractModuleAndBitcode(KInfo);

    auto Hash = Proteus.getStaticHash(KInfo);
    LOG_INFO("Hash value is {}", Hash.getValue());

    auto RecordAction = DB.takeSnapshot<VendorTypes>(
        PM->getVAStart(), PM->getTotalVASize(), KInfo, AllocatedBlobs, GridDim,
        BlockDim, Args, SharedMem, Stream);
    if (RecordAction)
      LOG_INFO("Successfully Recorded Prologue of Kernel {} NAME:{} GRID:({}, "
               "{}, {}) "
               "BLOCK:({}, {}, "
               "{}) SHM_SIZE:{}",
               func, KInfo.getName(), GridDim.x, GridDim.y, GridDim.z,
               BlockDim.x, BlockDim.y, BlockDim.z, SharedMem);
    auto ret =
        origLaunchKernel(func, GridDim, BlockDim, Args, SharedMem, Stream);
    if (RecordAction) {
      (*RecordAction)(KInfo.getBinaryInfo().getVarNameToGlobalInfo(),
                      AllocatedBlobs, Args, Stream);
      LOG_INFO("Successfully Recorded Epilogue of Kernel {} NAME:{} GRID:({}, "
               "{}, {}) "
               "BLOCK:({}, {}, "
               "{}) SHM_SIZE:{}",
               func, KInfo.getName(), GridDim.x, GridDim.y, GridDim.z,
               BlockDim.x, BlockDim.y, BlockDim.z, SharedMem);
    }
    return ret;
  }

  DeviceError_t rtSetDevice(int deviceID) {
    auto ret = origSetDeviceID(deviceID);
    if (DeviceID == -1) {
      DeviceID = deviceID;
    } else if (PM == nullptr) {
      DeviceID = deviceID;
    } else if (DeviceID != -1 && PM != nullptr)
      LOG_CRITICAL("Setting Device ID although it already "
                   "set and memory is "
                   "allocated");
    return ret;
  }

  DeviceError_t rtGetDevice(int *deviceID) { return origGetDeviceID(deviceID); }

  MnemeRecorder() : ExtractedIR(true) {
    VAStartAddr = nullptr;
    VATotalSize = 0;
    rtLib = MnemeDeviceRT::getRTLib();
    proteusLib = dlopen("libproteus.so", RTLD_NOW);
    RecordReplayDir = DB.getDir();
    DeviceID = -1;

    // Redirect overloaded device runtime functions.
    reinterpret_cast<void *&>(proteusLaunchKernel) =
        dlsym(proteusLib, MnemeDeviceRT::getLaunchKernelFnName());
    assert(proteusLaunchKernel &&
           "Expected non-null proteus-kernel-launch function pointer");

    reinterpret_cast<void *&>(origLaunchKernel) =
        dlsym(rtLib, MnemeDeviceRT::getLaunchKernelFnName());
    assert(origLaunchKernel &&
           "Expected non-null kernel-launch function pointer");

    reinterpret_cast<void *&>(origMallocDevice) =
        dlsym(rtLib, MnemeDeviceRT::getDeviceMallocFnName());
    assert(origMallocDevice &&
           "Expected non-null device malloc function pointer");

    reinterpret_cast<void *&>(origMallocPinned) =
        dlsym(rtLib, MnemeDeviceRT::getPinnedMallocFnName());
    assert(origMallocPinned &&
           "Expected non-null pinned malloc function pointer");

    reinterpret_cast<void *&>(origMallocManaged) =
        dlsym(rtLib, MnemeDeviceRT::getManagedMallocFnName());
    assert(origMallocManaged &&
           "Expected non-null managed malloc function pointer");

    reinterpret_cast<void *&>(origFreeHost) =
        dlsym(rtLib, MnemeDeviceRT::getPinnedFreeFnName());
    assert(origFreeHost && "Expected non-null Free Pinned Function");

    reinterpret_cast<void *&>(origFreeDevice) =
        dlsym(rtLib, MnemeDeviceRT::getDeviceFreeFnName());
    assert(origFreeDevice && "Expected non-null Device free function pointer");

    reinterpret_cast<void *&>(origSetDeviceID) =
        dlsym(rtLib, MnemeDeviceRT::getDeviceSetIDFnName());
    assert(origSetDeviceID && "Expected non-null set device id fn name");

    reinterpret_cast<void *&>(origGetDeviceID) =
        dlsym(rtLib, MnemeDeviceRT::getDeviceGetIDFnName());
    assert(origGetDeviceID && "Expected non-null get device id fn name");
  }

  ~MnemeRecorder() {
    if (PM)
      PM.reset();
    return;
    LOG_DEBUG("Releasing memory");
  }
};
} // namespace mneme
