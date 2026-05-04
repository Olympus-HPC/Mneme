#pragma once

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeConfig.hpp"
#include "mneme/MnemeKernelInfo.hpp"
#include "mneme/MnemeLLVMUtils.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeMemory.hpp"
#include "mneme/MnemePageManager.hpp"
#include "mneme/MnemeSnapshot.hpp"
#include "mneme/MnemeUtils.hpp"

#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <dlfcn.h>
#include <memory>
#include <utility>

#include <llvm/ADT/DenseMap.h>

#include <proteus/KernelMetadata.h>

namespace mneme {

template <DeviceVendors VendorTypes> struct RecorderRuntimeFunctions {
  using MnemeDeviceRT = DeviceTraits<VendorTypes>;
  using DeviceError_t = typename MnemeDeviceRT::DeviceError_t;
  using DeviceStream_t = typename MnemeDeviceRT::DeviceStream_t;

  void *rtLib = nullptr;

  DeviceError_t (*origLaunchKernel)(const void *func, dim3 gridDim,
                                    dim3 blockDim, void **args,
                                    size_t sharedMem,
                                    DeviceStream_t stream) = nullptr;
  DeviceError_t (*origMallocDevice)(void **ptr, size_t size) = nullptr;
  DeviceError_t (*origMallocPinned)(void **ptr, size_t size,
                                    unsigned int flags) = nullptr;
  DeviceError_t (*origMallocManaged)(void **ptr, size_t size,
                                     unsigned int flags) = nullptr;
  DeviceError_t (*origFreeDevice)(void *devPtr) = nullptr;
  DeviceError_t (*origFreeHost)(void *ptr) = nullptr;
  DeviceError_t (*origSetDeviceID)(int id) = nullptr;
  DeviceError_t (*origGetDeviceID)(int *id) = nullptr;

  RecorderRuntimeFunctions() {
    rtLib = MnemeDeviceRT::getRTLib();

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
};

template <DeviceVendors VendorTypes> class RecorderBackend {
public:
  using MnemeDeviceRT = DeviceTraits<VendorTypes>;
  using DeviceError_t = typename MnemeDeviceRT::DeviceError_t;
  using DeviceStream_t = typename MnemeDeviceRT::DeviceStream_t;

  virtual ~RecorderBackend() = default;

  virtual bool setMetadataForPointer(const void *ptr, Metadata md) = 0;
  virtual bool getMetadataForPointer(const void *ptr, Metadata &md) const = 0;
  virtual bool eraseMetadataForPointer(const void *ptr) = 0;

  virtual DeviceError_t rtMalloc(void **ptr, size_t size) = 0;
  virtual DeviceError_t rtManagedMalloc(void **ptr, size_t size,
                                        unsigned int flags) = 0;
  virtual DeviceError_t rtHostMalloc(void **ptr, size_t size,
                                     unsigned int flags) = 0;
  virtual DeviceError_t rtFree(void *ptr) = 0;
  virtual DeviceError_t rtHostFree(void *ptr) = 0;
  virtual DeviceError_t rtLaunchKernel(const void *func, dim3 &GridDim,
                                       dim3 &BlockDim, void **Args,
                                       size_t SharedMem,
                                       DeviceStream_t Stream) = 0;
  virtual DeviceError_t rtSetDevice(int deviceID) = 0;
  virtual DeviceError_t rtGetDevice(int *deviceID) = 0;
};

template <DeviceVendors VendorTypes>
class NullRecorder final : public RecorderBackend<VendorTypes> {
  using Base = RecorderBackend<VendorTypes>;
  using DeviceError_t = typename Base::DeviceError_t;
  using DeviceStream_t = typename Base::DeviceStream_t;

  RecorderRuntimeFunctions<VendorTypes> Runtime;

public:
  bool setMetadataForPointer(const void *, Metadata) override { return false; }
  bool getMetadataForPointer(const void *, Metadata &) const override {
    return false;
  }
  bool eraseMetadataForPointer(const void *) override { return false; }

  DeviceError_t rtMalloc(void **ptr, size_t size) override {
    return Runtime.origMallocDevice(ptr, size);
  }

  DeviceError_t rtManagedMalloc(void **ptr, size_t size,
                                unsigned int flags) override {
    return Runtime.origMallocManaged(ptr, size, flags);
  }

  DeviceError_t rtHostMalloc(void **ptr, size_t size,
                             unsigned int flags) override {
    return Runtime.origMallocPinned(ptr, size, flags);
  }

  DeviceError_t rtFree(void *ptr) override {
    return Runtime.origFreeDevice(ptr);
  }

  DeviceError_t rtHostFree(void *ptr) override {
    return Runtime.origFreeHost(ptr);
  }

  DeviceError_t rtLaunchKernel(const void *func, dim3 &GridDim, dim3 &BlockDim,
                               void **Args, size_t SharedMem,
                               DeviceStream_t Stream) override {
    return Runtime.origLaunchKernel(func, GridDim, BlockDim, Args, SharedMem,
                                    Stream);
  }

  DeviceError_t rtSetDevice(int deviceID) override {
    return Runtime.origSetDeviceID(deviceID);
  }

  DeviceError_t rtGetDevice(int *deviceID) override {
    return Runtime.origGetDeviceID(deviceID);
  }
};

template <DeviceVendors VendorTypes>
class RecordingBackend final : public RecorderBackend<VendorTypes> {
  using Base = RecorderBackend<VendorTypes>;
  using MnemeDeviceRT = typename Base::MnemeDeviceRT;
  using DeviceError_t = typename Base::DeviceError_t;
  using DeviceStream_t = typename Base::DeviceStream_t;

  RecorderRuntimeFunctions<VendorTypes> Runtime;
  RecordDatabase DB;
  llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> AllocatedBlobs;
  std::unique_ptr<PageManager<VendorTypes>> PM;

  // NOTE: We only keep track of the first time we set the device id. Once we
  // create the allocator we assume that the allocations go to the same device.
  int DeviceID = -1;

  void initializePageManagerIfNeeded() {
    if (PM)
      return;

    // NOTE: We need this arch cause internally we initialize the device.
    // FIXME: We need to have a DeviceTrait function to initialize the GPU
    // and call it separately here. Let's do this on a separate PR.
    auto arch = MnemeDeviceRT::GetDeviceArch();
    LOG_DEBUG("Initializing system {}", arch);
    if (DeviceID == -1)
      Runtime.origGetDeviceID(&DeviceID);
    PM = initializePageManager<VendorTypes>(
        DeviceID, (void *)MnemeDeviceRT::getSuggestedAddr());
  }

public:
  bool setMetadataForPointer(const void *ptr, Metadata md) override {
    if (!ptr)
      return false;

    auto It = AllocatedBlobs.find(const_cast<void *>(ptr));
    if (It == AllocatedBlobs.end())
      return false;

    It->second.setMetadata(std::move(md));
    return true;
  }

  bool getMetadataForPointer(const void *ptr, Metadata &md) const override {
    if (!ptr)
      return false;

    auto It = AllocatedBlobs.find(const_cast<void *>(ptr));
    if (It == AllocatedBlobs.end())
      return false;

    md = It->second.getMetadata();
    return true;
  }

  bool eraseMetadataForPointer(const void *ptr) override {
    if (!ptr)
      return false;

    auto It = AllocatedBlobs.find(const_cast<void *>(ptr));
    if (It == AllocatedBlobs.end())
      return false;

    It->second.setMetadata(Metadata{});
    return true;
  }

  DeviceError_t rtMalloc(void **ptr, size_t size) override {
    initializePageManagerIfNeeded();

    auto [Addr, ReservedSize] = PM->allocateAddr(size, nullptr);
    MnemeMemoryBlob<VendorTypes> MemBlob(ReservedSize,
                                         reinterpret_cast<void *>(Addr), size);
    auto ret = MemBlob.map(reinterpret_cast<void *>(Addr), ReservedSize, size);
    *ptr = MemBlob.ptr();
    AllocatedBlobs.insert({*ptr, std::move(MemBlob)});
    LOG_DEBUG("Intercepted Device Malloc PTR:{} SIZE:{} ACTUALSIZE:{}", *ptr,
              size, ReservedSize);
    return ret;
  }

  DeviceError_t rtManagedMalloc(void **ptr, size_t size,
                                unsigned int flags) override {
    auto ret = Runtime.origMallocManaged(ptr, size, flags);
    LOG_DEBUG("Intercepted Managed Malloc PTR:{} SIZE:{}", *ptr, size);
    LOG_WARN("Will not be able to replay Kernels acessing:{}", *ptr);
    return ret;
  }

  DeviceError_t rtHostMalloc(void **ptr, size_t size,
                             unsigned int flags) override {
    auto ret = Runtime.origMallocPinned(ptr, size, flags);
    LOG_WARN("Intercepted Pinned|Host Malloc PTR:{} SIZE:{}", *ptr, size);
    return ret;
  }

  DeviceError_t rtFree(void *ptr) override {
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
  }

  DeviceError_t rtHostFree(void *ptr) override {
    auto ret = Runtime.origFreeHost(ptr);
    LOG_DEBUG("Free pinned address:{}", ptr);
    return ret;
  }

  DeviceError_t rtLaunchKernel(const void *func, dim3 &GridDim, dim3 &BlockDim,
                               void **Args, size_t SharedMem,
                               DeviceStream_t Stream) override {
    initializePageManagerIfNeeded();

    // NOTE: Here we do something conceptually different. We no longer go through
    // proteus. We call immediately the vendor launcher. Thus we avoid overheads from caching etc.
    LOG_DEBUG("Received OptionalKernel Info {}",
              (void *)Runtime.origLaunchKernel);
    auto OptionalKernelInfo = proteus::runtime::captureKernelMetadata(func);
    if (!OptionalKernelInfo) {
      LOG_DEBUG("Information for kernel  {} is not included", func);
      return Runtime.origLaunchKernel(func, GridDim, BlockDim, Args, SharedMem,
                                      Stream);
    }
    auto &KInfo = OptionalKernelInfo.value();
    LOG_DEBUG("Continue with {}", KInfo.getName());
    LOG_INFO("Hash value is {}", KInfo.getStaticHash());

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
    auto ret = Runtime.origLaunchKernel(func, GridDim, BlockDim, Args,
                                        SharedMem, Stream);
    if (RecordAction) {
      (*RecordAction)(AllocatedBlobs, Args, Stream);
      LOG_INFO("Successfully Recorded Epilogue of Kernel {} NAME:{} GRID:({}, "
               "{}, {}) "
               "BLOCK:({}, {}, "
               "{}) SHM_SIZE:{}",
               func, KInfo.getName(), GridDim.x, GridDim.y, GridDim.z,
               BlockDim.x, BlockDim.y, BlockDim.z, SharedMem);
    }
    return ret;
  }

  DeviceError_t rtSetDevice(int deviceID) override {
    auto ret = Runtime.origSetDeviceID(deviceID);
    if (DeviceID == -1) {
      DeviceID = deviceID;
    } else if (PM == nullptr) {
      DeviceID = deviceID;
    } else if (DeviceID != -1 && PM != nullptr) {
      LOG_CRITICAL("Setting Device ID although it already "
                   "set and memory is "
                   "allocated");
    }
    return ret;
  }

  DeviceError_t rtGetDevice(int *deviceID) override {
    return Runtime.origGetDeviceID(deviceID);
  }
};

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
    return Backend->setMetadataForPointer(ptr, std::move(md));
  }

  bool getMetadataForPointer(const void *ptr, Metadata &md) const {
    return Backend->getMetadataForPointer(ptr, md);
  }

  bool eraseMetadataForPointer(const void *ptr) {
    return Backend->eraseMetadataForPointer(ptr);
  }

  DeviceError_t rtMalloc(void **ptr, size_t size) {
    return Backend->rtMalloc(ptr, size);
  }

  DeviceError_t rtManagedMalloc(void **ptr, size_t size, unsigned int flags) {
    return Backend->rtManagedMalloc(ptr, size, flags);
  }

  DeviceError_t rtHostMalloc(void **ptr, size_t size, unsigned int flags) {
    return Backend->rtHostMalloc(ptr, size, flags);
  }

  DeviceError_t rtFree(void *ptr) { return Backend->rtFree(ptr); }

  DeviceError_t rtHostFree(void *ptr) { return Backend->rtHostFree(ptr); }

  DeviceError_t rtLaunchKernel(const void *func, dim3 &GridDim, dim3 &BlockDim,
                               void **Args, size_t SharedMem,
                               DeviceStream_t Stream) {
    return Backend->rtLaunchKernel(func, GridDim, BlockDim, Args, SharedMem,
                                   Stream);
  }

  DeviceError_t rtSetDevice(int deviceID) {
    return Backend->rtSetDevice(deviceID);
  }

  DeviceError_t rtGetDevice(int *deviceID) {
    return Backend->rtGetDevice(deviceID);
  }
};

} // namespace mneme
