#pragma once

#include "MnemeMemory.hpp"
#include "MnemePageManager.hpp"
#include "MnemeUtils.hpp"
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
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeSnapshot.hpp"
#include "mneme/MnemeSymbols.hpp"

namespace mneme {

struct FatBinaryWrapper_t {
  int Magic;
  int Version;
  const char *Binary;
  void **PrelinkedFatBins;
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

private:
  void extractIR() { static_cast<ImplT &>(*this).extractIR(); }

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
  void registerFatBinEnd(void *ptr) { origRegisterFatBinaryEnd(ptr); }

  void **registerFatBin(FatBinaryWrapper_t *fatbin) {
    void **Handle = origRegisterFatBinary(fatbin);
    LOG_DEBUG("Register Fatbin Returned handle {}", (void *)Handle);
    HandleToBin.insert({Handle, fatbin});
    HandleToGlobalSymbol.insert({Handle, {}});

    return Handle;
  }

  void registerVar(void **fatBinHandle, char *hostVar, char *deviceAddress,
                   const char *deviceName, int ext, size_t size, int constant,
                   int global) {
    LOG_INFO("Register Global Variable: {} SIZE:{}, CONSTANT:{} GLOBAL:{} ",
             deviceName, size, constant, global);

    origRegisterDeviceVar(fatBinHandle, hostVar, deviceAddress, deviceName, ext,
                          size, constant, global);
    if (!constant)
      HandleToGlobalSymbol[fatBinHandle].emplace_back(
          GlobalVarInfo(deviceName, hostVar, size));
    return;
  }

  void registerFunc(void **fatBinHandle, const char *hostFun, char *deviceFun,
                    const char *deviceName, int thread_limit, uint3 *tid,
                    uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize) {
    LOG_INFO("Register Function : {} with a thread_limit off {} ", deviceName,
             thread_limit);
    if (!HandleToBin.contains(fatBinHandle))
      LOG_FATAL("Handle container does not contain fatbin handle");
    std::shared_ptr<KernelInfo> KI = std::make_shared<KernelInfo>(
        fatBinHandle, (const void *)hostFun, deviceFun);
    KernelInfoMap.insert({(const void *)hostFun, KI});
    HandleToKernels[fatBinHandle].emplace_back(KI);
    origRegisterFunction(fatBinHandle, hostFun, deviceFun, deviceName,
                         thread_limit, tid, bid, bDim, gDim, wSize);
  };

  DeviceError_t rtMalloc(void **ptr, size_t size) {
    // TODO: Find a better way to find the current active device;
    int DeviceID = 0;

    std::call_once(ExtractFlag, [this]() {
      PM = initializePageManager<MnemeDeviceRT>();
      extractIR();
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
    if (!KernelInfoMap.contains(func)) {
      LOG_WARN("Skipping kernel cause not tracked in map");
      return origLaunchKernel(func, GridDim, BlockDim, Args, SharedMem, Stream);
    }

    if (BlackList.contains(func)) {
      LOG_WARN("Skipping kernel cause kernel is blacklisted");
      return origLaunchKernel(func, GridDim, BlockDim, Args, SharedMem, Stream);
    }

    std::call_once(ExtractFlag, [this]() {
      PM = initializePageManager<MnemeDeviceRT>();
      extractIR();
      getGlobalAddresses();
    });

    auto KInfo = KernelInfoMap[func];
    auto Handle = KInfo->getHandle();
    if (!HandleToGlobalSymbol.contains(Handle))
      LOG_FATAL("Accessing Kernel Without a Handle");

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

  std::pair<llvm::stable_hash, std::string> storeModule(llvm::Module &M) {
    static int TotalModules = 0;
    std::error_code EC;

    std::string StrBuffer;
    llvm::raw_string_ostream RSO(StrBuffer);

    // Serialize the module into the string buffer
    llvm::WriteBitcodeToFile(M, RSO);
    RSO.flush();

    uint64_t StableHash = llvm::stable_hash_combine_string(
        llvm::StringRef(StrBuffer.data(), StrBuffer.size()));

    std::string Filename(std::filesystem::path(
                             llvm::Twine(RecordReplayDir + "/RecordedIR_" +
                                         std::to_string(TotalModules++) + ".bc")
                                 .str())
                             .string());
    llvm::raw_fd_ostream OutBC(Filename, EC);
    if (EC)
      LOG_FATAL("Cannot write module ir file");

    OutBC << StrBuffer;
    LOG_DEBUG("Stored Module with StaticHash:{} to file {}", StableHash,
              std::filesystem::canonical(Filename).string());
    OutBC.close();
    return std::make_pair(StableHash,
                          std::filesystem::canonical(Filename).string());
  }

  MnemeRecorder() : ExtractedIR(true) {
    VAStartAddr = nullptr;
    VATotalSize = 0;
    rtLib = ImplT::getRTLib();
    RecordReplayDir = DB.getDir();
    // MemManager = nullptr;

    // Redirect overloaded device runtime functions.
    reinterpret_cast<void *&>(origLaunchKernel) =
        dlsym(rtLib, ImplT::getLaunchKernelFnName());
    assert(origLaunchKernel &&
           "Expected non-null kernel-launch function pointer");

    reinterpret_cast<void *&>(origMallocDevice) =
        dlsym(rtLib, ImplT::getDeviceMallocFnName());
    assert(origMallocDevice &&
           "Expected non-null device malloc function pointer");

    reinterpret_cast<void *&>(origMallocPinned) =
        dlsym(rtLib, ImplT::getPinnedMallocFnName());
    assert(origMallocPinned &&
           "Expected non-null pinned malloc function pointer");

    reinterpret_cast<void *&>(origMallocManaged) =
        dlsym(rtLib, ImplT::getManagedMallocFnName());
    assert(origMallocManaged &&
           "Expected non-null managed malloc function pointer");

    reinterpret_cast<void *&>(origFreeHost) =
        dlsym(rtLib, ImplT::getPinnedFreeFnName());
    assert(origFreeHost && "Expected non-null Free Pinned Function");

    reinterpret_cast<void *&>(origFreeDevice) =
        dlsym(rtLib, ImplT::getDeviceFreeFnName());
    assert(origFreeDevice && "Expected non-null Device free function pointer");

    reinterpret_cast<void *&>(origRegisterFunction) =
        dlsym(rtLib, ImplT::getUURegisterFunctionFnName());
    assert(origRegisterFunction && "Expected non-null Register Function");

    reinterpret_cast<void *&>(origRegisterDeviceVar) =
        dlsym(rtLib, ImplT::getUURegisterVarFnName());
    assert(origRegisterDeviceVar && "Expected non-null register Device Var");

    reinterpret_cast<void *&>(origRegisterDeviceVar) =
        dlsym(rtLib, ImplT::getUURegisterVarFnName());
    assert(origRegisterDeviceVar && "Expected non-null register Device Var");

    reinterpret_cast<void *&>(origRegisterFatBinary) =
        dlsym(rtLib, ImplT::getUURegisterFatbinFnName());
    assert(origRegisterFatBinary && "Expected non-null register Device Var");

    if (ImplT::hasFatBinEnd) {
      assert(origRegisterDeviceVar && "Expected non-null register Device Var");
    }
  }

  ~MnemeRecorder() {
    MnemeDeviceRT::freeVirtualAddress(PM->getVAStart(), PM->getTotalVASize());
  }
};
} // namespace mneme
