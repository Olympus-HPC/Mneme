#pragma once

#include "Logger.hpp"
#include "MnemeMemory.hpp"
#include "MnemePageManager.hpp"
#include "Utils.hpp"
#include <assert.h>
#include <cstddef>
#include <dlfcn.h>

#include "llvm/Support/raw_ostream.h"
#include <filesystem>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StableHashing.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <mutex>

#include "DeviceTraits.hpp"
#include "MnemeSnapshot.hpp"
#include "MnemeSymbols.hpp"
#include "llvm/Bitcode/BitcodeWriter.h"

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
        DBG(Logger::logs("mneme")
                << "GlobalVar: " << GVar.Name << " DevAddr: " << GVar.DevAddr
                << " HostSymbolAddr " << GVar.HostSymbolAddr
                << " Size: " << std::dec << GVar.VarSize << "\n";)
      }
    }
  }

public:
  void registerFatBinEnd(void *ptr) {
    DBG(Logger::logs("mneme") << "Registering FatBinaryEnd at address "
                              << std::hex << ptr << std::dec << "\n");
    origRegisterFatBinaryEnd(ptr);
  }

  void **registerFatBin(FatBinaryWrapper_t *fatbin) {
    void **Handle = origRegisterFatBinary(fatbin);
    DBG(Logger::logs("mneme")
        << "Registering FatBinary at address " << std::hex << fatbin << std::dec
        << " Return Ptr is: " << std::hex << Handle << std::dec << "\n");
    HandleToBin.insert({Handle, fatbin});
    for (auto &[H, B] : HandleToBin) {
      DBG(Logger::logs("mneme")
              << "Handle : " << std::hex << H << std::dec << " mapped to "
              << std::hex << B << std::dec << "\n";)
    }
    Logger::logs("mneme") << "Add of this is " << std::hex << this << std::dec
                          << "\n";

    HandleToGlobalSymbol.insert({Handle, {}});

    return Handle;
  }

  void registerVar(void **fatBinHandle, char *hostVar, char *deviceAddress,
                   const char *deviceName, int ext, size_t size, int constant,
                   int global) {
    DBG(Logger::logs("mneme")
        << "Registering variable from handle " << std::hex << fatBinHandle
        << std::dec << " " << hostVar << "In address" << deviceName << " "
        << ext << " " << size << " " << constant << " " << global << "\n");
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
    DBG(Logger::logs("mneme")
        << "Registering Function from handle " << std::hex << fatBinHandle
        << std::dec << " HostFun:" << hostFun << " deviceFun:" << deviceFun
        << " deviceName:" << deviceName << " thread_limit:" << thread_limit
        << "\n");
    Logger::logs("mneme") << "Add of this is " << std::hex << this << std::dec
                          << "\n";
    if (!HandleToBin.contains(fatBinHandle))
      FATAL_ERROR("Handle container does not contain fatbin handle");
    std::shared_ptr<KernelInfo> KI =
        std::make_shared<KernelInfo>(fatBinHandle, deviceFun);
    KernelInfoMap.insert({(const void *)hostFun, KI});
    HandleToKernels[fatBinHandle].emplace_back(KI);
    origRegisterFunction(fatBinHandle, hostFun, deviceFun, deviceName,
                         thread_limit, tid, bid, bDim, gDim, wSize);
  };

  DeviceError_t rtMalloc(void **ptr, size_t size) {
    MnemeMemoryBlob<VendorTypes> MemBlob;
    auto ret = MemBlob.allocate(0, size);
    *ptr = MemBlob.ptr();
    AllocatedBlobs.insert({*ptr, std::move(MemBlob)});
    DBG(Logger::logs("mneme") << "Malloced Device Pointer " << *ptr
                              << " with size: " << size << "\n");
    return ret;
  };

  DeviceError_t rtManagedMalloc(void **ptr, size_t size, unsigned int flags) {
    auto ret = origMallocManaged(ptr, size, flags);
    DBG(Logger::logs("mneme") << "Malloced Managed Pointer " << *ptr
                              << " with size: " << size << "\n");
    return ret;
  };

  DeviceError_t rtHostMalloc(void **ptr, size_t size, unsigned int flags) {
    auto ret = origMallocPinned(ptr, size, flags);
    DBG(Logger::logs("mneme") << "Malloced Pinned Pointer " << *ptr
                              << " with size: " << size << "\n");
    return ret;
  }

  DeviceError_t rtFree(void *ptr) {
    if (!AllocatedBlobs.contains(ptr))
      FATAL_ERROR("Free address that is not being allocated through Mneme\n");
    auto ret = AllocatedBlobs[ptr].release();
    AllocatedBlobs.erase(ptr);
    DBG(Logger::logs("mneme")
        << "Free Address " << std::hex << ptr << std::dec << "\n");
    return ret;
  };

  DeviceError_t rtHostFree(void *ptr) {
    auto ret = origFreeHost(ptr);
    DBG(Logger::logs("mneme")
        << "Free pinned address: " << std::hex << ptr << std::dec << "\n");
    return ret;
  }

  DeviceError_t rtLaunchKernel(const void *func, dim3 &GridDim, dim3 &BlockDim,
                               void **Args, size_t SharedMem,
                               DeviceStream_t Stream) {
    if (!KernelInfoMap.contains(func)) {
      Logger::warn() << "Kernel not included in Map, skippping ...\n";
      return origLaunchKernel(func, GridDim, BlockDim, Args, SharedMem, Stream);
    }

    std::call_once(ExtractFlag, [this]() {
      extractIR();
      getGlobalAddresses();
    });

    auto KInfo = KernelInfoMap[func];
    auto Handle = KInfo->getHandle();
    if (!HandleToGlobalSymbol.contains(Handle))
      FATAL_ERROR("Accessing Kernel Without a Handle");

    auto RecordAction = DB.takeSnapshot<VendorTypes>(
        KInfo, HandleToGlobalSymbol[Handle], AllocatedBlobs, GridDim, BlockDim,
        Args, SharedMem, Stream);
    DBG(Logger::logs("mneme")
            << "Launching Kernel " << std::hex << func << std::dec << " KName"
            << KInfo->Name << " GDimX: " << GridDim.x << " GDimY: " << GridDim.y
            << " GDimZ: " << GridDim.z << " BDimX: " << BlockDim.x
            << " BDimY: " << BlockDim.y << " BDimZ: " << BlockDim.z << "\n";);
    auto ret =
        origLaunchKernel(func, GridDim, BlockDim, Args, SharedMem, Stream);
    if (RecordAction)
      (*RecordAction)(HandleToGlobalSymbol[Handle], AllocatedBlobs, Args,
                      Stream);
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

    std::string Filename(
        std::filesystem::path(llvm::Twine(RecordReplayDir + "RecordedIR_" +
                                          std::to_string(TotalModules) + ".bc")
                                  .str())
            .string());
    llvm::raw_fd_ostream OutBC(Filename, EC);
    if (EC)
      FATAL_ERROR("Cannot write module ir file");

    OutBC << StrBuffer;
    DBG(std::cout << "Registered Record replay descr\n");
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
};
} // namespace mneme
