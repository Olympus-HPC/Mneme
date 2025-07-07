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

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeDeviceBinary.hpp"
#include "mneme/MnemeKernelInfo.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeSnapshot.hpp"
#include "mneme/MnemeSymbols.hpp"

namespace mneme {

struct MnemeDeviceExecutable {
  /* An execution can have a collection of Linked  Binaries. Without RDC enabled
   * every translation unit is handled as a separate linked binary and the
   * scoped is limited in that binary */
  llvm::LLVMContext Ctx;
  llvm::DenseMap<DeviceHandle, MnemeDeviceLinkedBin> LinkedBinaries;
  llvm::DenseMap<void *, const char *> PendingRegistrations;
  llvm::DenseMap<const void *, std::shared_ptr<KernelInfo>> TrackedKernels;
  DeviceHandle CurrHandle;
  MnemeDeviceExecutable() : CurrHandle(nullptr) {}
};

template <typename ImplT, DeviceVendors VendorTypes> class MnemeRecorder {
protected:
  void *rtLib;
  std::string RecordReplayDir;
  llvm::DenseMap<void **, FatBinaryWrapper_t *> HandleToBin;
  llvm::DenseMap<void **, llvm::SmallVector<std::shared_ptr<KernelInfo>>>
      HandleToKernels;
  llvm::DenseMap<const void *, std::shared_ptr<KernelInfo>> KernelInfoMap;
  llvm::DenseMap<void **, llvm::SmallVector<GlobalVarInfo>>
      HandleToGlobalSymbol;
  llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> AllocatedBlobs;
  llvm::DenseSet<const void *> BlackList;
  MnemeDeviceExecutable Executable;

  std::unique_ptr<PageManager> PM;
  void *VAStartAddr;
  int64_t VATotalSize;

public:
  using MnemeDeviceRT = DeviceTraits<VendorTypes>;
  using DeviceError_t = typename MnemeDeviceRT::DeviceError_t;
  using DeviceStream_t = typename MnemeDeviceRT::DeviceStream_t;
  using KernelFunction_t = typename MnemeDeviceRT::KernelFunction_t;

private:
  bool ExtractedIR;
  RecordDatabase DB;
  std::once_flag ExtractFlag;
  bool ReleaseMemory;

  DeviceError_t (*origLaunchKernel)(const void *func, dim3 gridDim,
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

  void (*origRegisterDeviceVar)(void **fatbinHandle, char *hostVar,
                                char *deviceAddress, const char *deviceName,
                                int ext, size_t size, int constant, int global);

  void (*origRegisterFunction)(void **fatbinHandle, const char *hostFun,
                               char *deviceFun, const char *deviceName,
                               int thread_limit, uint3 *tid, uint3 *bid,
                               dim3 *bDim, dim3 *gDim, int *wSize);

  void **(*origRegisterFatBinary)(void *fatDevbin);

  void (*origRegisterFatBinaryEnd)(void *);

  void (*origUnregisterFatBinary)(void **);

private:
  void getGlobalAddresses() {
    for (auto &[Handle, GVars] : HandleToGlobalSymbol) {
      for (auto &GVar : GVars) {
        static_cast<ImplT &>(*this).initializeGlobal(GVar);
        LOG_INFO("Getting Global Variable: {} stored at address {} mapped "
                 "with host symbol addr {} of size {}",
                 GVar.Name, GVar.DevAddr, GVar.HostSymbolAddr, GVar.VarSize);
      }
    }
  }

public:
  void unregisterFatBinEnd(void **ptr) {
    LOG_DEBUG("Unregistering fatbinary");
    if (!ReleaseMemory) {
      LOG_DEBUG("Releasing memory");
      MnemeDeviceRT::freeVirtualAddress(PM->getVAStart(), PM->getTotalVASize());
      LOG_DEBUG("Done with Releasing memory");
      ReleaseMemory = true;
    }
    origUnregisterFatBinary(ptr);
  }

  void registerFatBinEnd(void *ptr) {
    LOG_DEBUG("Marking end of fat binary");
    origRegisterFatBinaryEnd(ptr);
  }

  void **registerFatBin(FatBinaryWrapper_t *fatbin) {
    void **Handle = origRegisterFatBinary(fatbin);
    LOG_DEBUG("Register Fatbin Returned handle {}", (void *)Handle);
    HandleToBin.insert({Handle, fatbin});
    HandleToGlobalSymbol.insert({Handle, {}});

    return Handle;
  }

  void explicitRegisterPreLinkedBinary(FatBinaryWrapper_t *FatbinWrapper,
                                       const char *ModuleId) {
    LOG_DEBUG("Received prelinked binary with the following fields {} {} {} {}",
              FatbinWrapper->Magic, FatbinWrapper->Version,
              (void *)FatbinWrapper->Binary,
              (void *)FatbinWrapper->PrelinkedFatBins);
    Executable.PendingRegistrations.insert(
        {(void *)FatbinWrapper->Binary, ModuleId});
  }

  void explicitRegisterFatBin(DeviceHandle Handle,
                              FatBinaryWrapper_t *FatbinWrapper,
                              const char *ModuleId) {
    if (Executable.CurrHandle == nullptr)
      Executable.CurrHandle = Handle;
    if (Executable.CurrHandle != Handle) {
#ifdef MNEME_ENABLE_CUDA
      LOG_FATAL("Current Handle is still open and we received a new one");
#endif
      Executable.CurrHandle = Handle;
    }

    if (Executable.LinkedBinaries.contains(Handle)) {
      LOG_FATAL("Received a new Handle but we already have a fatbinary for "
                "this handle");
    }

    auto [LinkedIt, inserted] =
        Executable.LinkedBinaries.try_emplace(Handle, Handle, FatbinWrapper);
    auto &Linked = LinkedIt->second;
    // TODO: Do we need this for HIP?
    if (FatbinWrapper->Version == 1) {
      Linked.ModuleIds.push_back(ModuleId);
    } else if (FatbinWrapper->Version == 2) {
      // NOTE: On version 2 binaries, we should not pushs the module-id of this
      // Fatbinary. Instead we inherit the module ids from the internal
      // fat-binaries.
      for (int I = 0; FatbinWrapper->PrelinkedFatBins[I] != nullptr; I++) {
        auto it = Executable.PendingRegistrations.find(
            FatbinWrapper->PrelinkedFatBins[I]);
        if (it != Executable.PendingRegistrations.end()) {
          // Element exists
          auto &val = it->second;
          Linked.ModuleIds.push_back(val);
          Executable.PendingRegistrations.erase(it);
        } else {
          Linked.ModuleIds.push_back("");
        }
        LOG_DEBUG("RDC Binary {} includes Binary address of {}", I,
                  (void *)FatbinWrapper->PrelinkedFatBins[I]);
      }
    } else {
      LOG_FATAL("Cannot handle binary type {}", FatbinWrapper->Version);
    }
  }

  void explicitEndRegisterFatBinary(DeviceHandle Handle) {

    if (Executable.CurrHandle != Handle) {
      LOG_WARN("Register Fat Binary End Handle does not match with current "
               "Handle {}:{}",
               (void *)Handle, (void *)Executable.CurrHandle);
    }
    // We turn off the tracking of the current handle.
    Executable.CurrHandle = nullptr;
  }

  void registerVar(DeviceHandle Handle, char *hostVar, char *deviceAddress,
                   const char *deviceName, int ext, size_t size, int constant,
                   int global) {
    LOG_INFO("Register Global Variable: {} SIZE:{}, CONSTANT:{} GLOBAL:{} ",
             deviceName, size, constant, global);

    origRegisterDeviceVar(Handle, hostVar, deviceAddress, deviceName, ext, size,
                          constant, global);

    auto it = Executable.LinkedBinaries.find(Handle);
    if (it == Executable.LinkedBinaries.end()) {
      LOG_DEBUG("Dropping tracking of {} as we are not tracking the fatbinary",
                deviceName);
      return;
    }

    if (constant)
      return;

    it->second.GlobalSymbols.emplace_back(
        GlobalVarInfo(deviceName, hostVar, size));
    return;
  }

  void registerFunc(DeviceHandle Handle, const char *hostFun, char *deviceFun,
                    const char *deviceName, int thread_limit, uint3 *tid,
                    uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize) {
    LOG_INFO("Register Function with handle: {} and Name: {} with a "
             "thread_limit off {} ",
             (void *)Handle, deviceName, thread_limit);

    auto it = Executable.LinkedBinaries.find(Handle);
    if (it == Executable.LinkedBinaries.end())
      LOG_FATAL("Handle container does not contain fatbin handle");

    auto &LinkedBin = it->second;

    std::shared_ptr<KernelInfo> KI = std::make_shared<KernelInfo>(
        LinkedBin, (const void *)hostFun, deviceFun);
    Executable.TrackedKernels.insert({(const void *)hostFun, KI});

    origRegisterFunction(Handle, hostFun, deviceFun, deviceName, thread_limit,
                         tid, bid, bDim, gDim, wSize);
  };

  DeviceError_t rtMalloc(void **ptr, size_t size) {
    // TODO: Find a better way to find the current active device;
    int DeviceID = 0;

    std::call_once(ExtractFlag, [this]() {
      PM = initializePageManager<MnemeDeviceRT>();
      getGlobalAddresses();
    });

    auto [Addr, ReservedSize] = PM->allocateAddr(size, nullptr);
    MnemeMemoryBlob<VendorTypes> MemBlob(
        ReservedSize, reinterpret_cast<void *>(Addr), size, DeviceID);
    auto ret = MemBlob.map(reinterpret_cast<void *>(Addr), ReservedSize, size,
                           DeviceID);
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
    auto it = Executable.TrackedKernels.find(func);

    if (it == Executable.TrackedKernels.end()) {
      LOG_WARN("Skipping kernel {} cause not tracked", func);
      return origLaunchKernel(func, GridDim, BlockDim, Args, SharedMem, Stream);
    }

    if (BlackList.contains(func)) {
      LOG_WARN("Skipping kernel cause kernel is blacklisted");
      return origLaunchKernel(func, GridDim, BlockDim, Args, SharedMem, Stream);
    }

    auto KInfo = it->second;
    std::call_once(ExtractFlag, [this]() {
      // NOTE: We need this arch cause internally we initialize the device.
      // FIXME: We need to have a DeviceTrait function to initialize the GPU
      // and call it separately here. Let's do this on a separate PR
      auto arch = MnemeDeviceRT::GetDeviceArch();
      LOG_DEBUG("Initializing system {}", arch);
      PM = initializePageManager<MnemeDeviceRT>();
      getGlobalAddresses();
    });

    auto &LinkedExecutable = KInfo->getHandle();
    if (LinkedExecutable.ExtractCode<VendorTypes>(Executable.Ctx)) {
      LinkedExecutable.StoreModules<VendorTypes>(RecordReplayDir);
      LinkedExecutable.FindKernels<VendorTypes>();
    }

    void **Handle = nullptr;

    auto RecordAction = DB.takeSnapshot<VendorTypes>(
        PM->getVAStart(), PM->getTotalVASize(), KInfo,
        HandleToGlobalSymbol[Handle], AllocatedBlobs, GridDim, BlockDim, Args,
        SharedMem, Stream);
    if (RecordAction)
      LOG_INFO("Successfully Recorded Prologue of Kernel {} NAME:{} GRID:({}, "
               "{}, {}) "
               "BLOCK:({}, {}, "
               "{}) SHM_SIZE:{}",
               func, KInfo->Name, GridDim.x, GridDim.y, GridDim.z, BlockDim.x,
               BlockDim.y, BlockDim.z, SharedMem);
    auto ret =
        origLaunchKernel(func, GridDim, BlockDim, Args, SharedMem, Stream);
    if (RecordAction) {
      (*RecordAction)(HandleToGlobalSymbol[Handle], AllocatedBlobs, Args,
                      Stream);
      LOG_INFO("Successfully Recorded Epilogue of Kernel {} NAME:{} GRID:({}, "
               "{}, {}) "
               "BLOCK:({}, {}, "
               "{}) SHM_SIZE:{}",
               func, KInfo->Name, GridDim.x, GridDim.y, GridDim.z, BlockDim.x,
               BlockDim.y, BlockDim.z, SharedMem);
    }
    return ret;
  }

  MnemeRecorder() : ExtractedIR(true), ReleaseMemory(false) {
    VAStartAddr = nullptr;
    VATotalSize = 0;
    rtLib = MnemeDeviceRT::getRTLib();
    RecordReplayDir = DB.getDir();
    // MemManager = nullptr;

    // Redirect overloaded device runtime functions.
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

    reinterpret_cast<void *&>(origRegisterFunction) =
        dlsym(rtLib, MnemeDeviceRT::getUURegisterFunctionFnName());
    assert(origRegisterFunction && "Expected non-null Register Function");

    reinterpret_cast<void *&>(origRegisterDeviceVar) =
        dlsym(rtLib, MnemeDeviceRT::getUURegisterVarFnName());
    assert(origRegisterDeviceVar && "Expected non-null register Device Var");

    reinterpret_cast<void *&>(origRegisterDeviceVar) =
        dlsym(rtLib, MnemeDeviceRT::getUURegisterVarFnName());
    assert(origRegisterDeviceVar && "Expected non-null register Device Var");

    reinterpret_cast<void *&>(origRegisterFatBinary) =
        dlsym(rtLib, MnemeDeviceRT::getUURegisterFatbinFnName());
    assert(origRegisterFatBinary && "Expected non-null register Device Var");

    reinterpret_cast<void *&>(origUnregisterFatBinary) =
        dlsym(rtLib, MnemeDeviceRT::getUUUnRegisterFatBinaryFnName());
    assert(origUnregisterFatBinary && "Expected non-null unregister fatbinary");

    if (MnemeDeviceRT::hasFatBinEnd) {
      reinterpret_cast<void *&>(origRegisterFatBinaryEnd) =
          dlsym(rtLib, MnemeDeviceRT::getUURegisterFatbinEndFnName());
      assert(origRegisterFatBinaryEnd &&
             "Expected non-null register Device Var");
    }
  }

  ~MnemeRecorder() {
    if (ReleaseMemory)
      return;
    LOG_DEBUG("Releasing memory");
    MnemeDeviceRT::freeVirtualAddress(PM->getVAStart(), PM->getTotalVASize());
    LOG_DEBUG("Done with Releasing memory");
    ReleaseMemory = true;
  }
};
} // namespace mneme
