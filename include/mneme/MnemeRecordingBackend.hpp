#pragma once

#include "mneme/MnemeKernelInfo.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeMemory.hpp"
#include "mneme/MnemePageManager.hpp"
#include "mneme/MnemeRecorderBackend.hpp"
#include "mneme/MnemeSnapshot.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <utility>

#include <llvm/ADT/DenseMap.h>

#include <proteus/KernelMetadata.h>

namespace mneme {

template <DeviceVendors VendorTypes>
class RecordingBackend final : public RecorderBackend<VendorTypes> {
  using Base = RecorderBackend<VendorTypes>;
  using MnemeDeviceRT = typename Base::MnemeDeviceRT;
  using DeviceError_t = typename Base::DeviceError_t;
  using DeviceStream_t = typename Base::DeviceStream_t;

  RecorderRuntimeFunctions<VendorTypes> Runtime;
  RecordDatabase DB;
  llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> AllocatedBlobs;
  std::map<std::uintptr_t, void *> AllocationIndex;
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

  static std::size_t builtinByteWidth(BuiltinDType builtin) {
    switch (builtin) {
    case BuiltinDType::U8:
    case BuiltinDType::I8:
      return 1;
    case BuiltinDType::U16:
    case BuiltinDType::I16:
    case BuiltinDType::F16:
      return 2;
    case BuiltinDType::U32:
    case BuiltinDType::I32:
    case BuiltinDType::F32:
      return 4;
    case BuiltinDType::U64:
    case BuiltinDType::I64:
    case BuiltinDType::F64:
      return 8;
    }
    return 1;
  }

  typename llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>>::iterator
  findOwningBlob(const void *ptr) {
    if (!ptr)
      return AllocatedBlobs.end();

    auto Addr = reinterpret_cast<std::uintptr_t>(ptr);
    auto It = AllocationIndex.upper_bound(Addr);
    if (It == AllocationIndex.begin())
      return AllocatedBlobs.end();

    --It;
    auto BlobIt = AllocatedBlobs.find(It->second);
    if (BlobIt == AllocatedBlobs.end())
      return AllocatedBlobs.end();

    auto Base = reinterpret_cast<std::uintptr_t>(BlobIt->first);
    auto Size = BlobIt->second.getSize();
    if (Addr < Base || (Addr - Base) >= Size)
      return AllocatedBlobs.end();
    return BlobIt;
  }

public:
  bool setMetadataForPointer(const void *ptr, Metadata md) override {
    if (!ptr) {
      LOG_WARN("setMetadataForPointer called with null pointer");
      return false;
    }

    auto It = AllocatedBlobs.find(const_cast<void *>(ptr));
    if (It == AllocatedBlobs.end()) {
      LOG_DEBUG("exact-match miss ptr={} tag={} allocated_blobs={}", ptr,
                md.tag.value_or("no_tag"), AllocatedBlobs.size());
      LOG_WARN(
          "annotation dropped after exact-match miss ptr={} tag={} allocated_blobs={}",
          ptr, md.tag.value_or("no_tag"), AllocatedBlobs.size());
      return false;
    }

    LOG_INFO(
        "exact-match hit ptr={} size={} actual_size={} builtin={} threshold={} threshold_kind={} norm={} tag={}",
        ptr, It->second.getSize(), It->second.getActualSize(),
        static_cast<unsigned>(md.builtin), md.threshold,
        static_cast<unsigned>(md.threshold_kind), static_cast<unsigned>(md.norm),
        md.tag.value_or("no_tag"));
    It->second.setMetadata(std::move(md));
    return true;
  }

  bool setMetadataForRegion(const void *ptr, size_t bytes, Metadata md) override {
    if (!ptr) {
      LOG_WARN("setMetadataForRegion called with null pointer");
      return false;
    }

    if (bytes == 0) {
      LOG_WARN("setMetadataForRegion called with zero bytes ptr={} tag={}",
               ptr, md.tag.value_or("no_tag"));
      return false;
    }

    auto It = findOwningBlob(ptr);
    if (It == AllocatedBlobs.end()) {
      LOG_WARN(
          "region annotation could not resolve owning allocation ptr={} bytes={} tag={} allocated_blobs={}",
          ptr, bytes, md.tag.value_or("no_tag"), AllocatedBlobs.size());
      return false;
    }

    auto Base = reinterpret_cast<std::uintptr_t>(It->first);
    auto Addr = reinterpret_cast<std::uintptr_t>(ptr);
    auto Offset = static_cast<uint64_t>(Addr - Base);
    auto BlobSize = It->second.getSize();
    if (bytes > BlobSize || Offset > BlobSize - bytes) {
      LOG_WARN(
          "region annotation exceeds owning allocation ptr={} base_ptr={} offset={} bytes={} blob_size={} tag={}",
          ptr, It->first, Offset, bytes, BlobSize, md.tag.value_or("no_tag"));
      return false;
    }

    auto ElemSize = builtinByteWidth(md.builtin);
    if ((bytes % ElemSize) != 0) {
      LOG_WARN(
          "region annotation byte extent is not aligned to builtin type ptr={} bytes={} builtin={} elem_size={} tag={}",
          ptr, bytes, static_cast<unsigned>(md.builtin), ElemSize,
          md.tag.value_or("no_tag"));
      return false;
    }

    if (!It->second.setRegionMetadata(Offset, bytes, md)) {
      LOG_WARN(
          "region annotation rejected due to overlap ptr={} base_ptr={} offset={} bytes={} tag={}",
          ptr, It->first, Offset, bytes, md.tag.value_or("no_tag"));
      return false;
    }

    LOG_INFO(
        "region annotation registered ptr={} base_ptr={} offset={} bytes={} builtin={} threshold={} threshold_kind={} norm={} tag={}",
        ptr, It->first, Offset, bytes, static_cast<unsigned>(md.builtin),
        md.threshold, static_cast<unsigned>(md.threshold_kind),
        static_cast<unsigned>(md.norm), md.tag.value_or("no_tag"));
    return true;
  }

  bool getMetadataForPointer(const void *ptr, Metadata &md) const override {
    if (!ptr) {
      LOG_WARN("getMetadataForPointer called with null pointer");
      return false;
    }

    auto It = AllocatedBlobs.find(const_cast<void *>(ptr));
    if (It == AllocatedBlobs.end()) {
      LOG_DEBUG("getMetadataForPointer exact-match miss ptr={}", ptr);
      LOG_DEBUG(
          "getMetadataForPointer returning false after exact-match miss ptr={}",
          ptr);
      return false;
    }

    md = It->second.getMetadata();
    LOG_DEBUG("getMetadataForPointer hit ptr={} tag={}", ptr,
              md.tag.value_or("no_tag"));
    return true;
  }

  bool eraseMetadataForPointer(const void *ptr) override {
    if (!ptr) {
      LOG_WARN("eraseMetadataForPointer called with null pointer");
      return false;
    }

    auto It = AllocatedBlobs.find(const_cast<void *>(ptr));
    if (It == AllocatedBlobs.end()) {
      LOG_DEBUG("eraseMetadataForPointer exact-match miss ptr={}", ptr);
      LOG_DEBUG(
          "eraseMetadataForPointer returning false after exact-match miss ptr={}",
          ptr);
      return false;
    }

    LOG_INFO("eraseMetadataForPointer hit ptr={} tag={}", ptr,
             It->second.getMetadata().tag.value_or("no_tag"));
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
    AllocationIndex[reinterpret_cast<std::uintptr_t>(*ptr)] = *ptr;
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
    AllocationIndex.erase(reinterpret_cast<std::uintptr_t>(ptr));
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

    // NOTE: Here we do something conceptually different. We no longer go
    // through proteus. We call immediately the vendor launcher. Thus we avoid
    // overheads from caching etc.
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
      // write kernel record incrementally after epilogue is recorded
      DB.writeKernelJSON(KInfo.getStaticHash());
    }
    return ret;
  }

  DeviceError_t rtSetDevice(int deviceID) override {
    auto ret = Runtime.origSetDeviceID(deviceID);
    if (DeviceID == -1) {
      DeviceID = deviceID;
    } else if (PM == nullptr) {
      DeviceID = deviceID;
    } else if (deviceID == DeviceID) {
      // no change; ignore
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

  ~RecordingBackend() {
    LOG_DEBUG("RecordingBackend destructor: releasing {} blobs", AllocatedBlobs.size());

    // Release all unreleased blobs before the DenseMap destructs
    for (auto &Entry : AllocatedBlobs) {
      auto &Blob = Entry.second;
      if (Blob.getBlobAddr() != nullptr) {
        LOG_DEBUG("Releasing blob at addr={} during recorder cleanup",
                  Blob.getBlobAddr());
        Blob.release();
      }
    }

    if (PM)
      PM.reset();

    LOG_DEBUG("RecordingBackend destructor complete");
  }
};

} // namespace mneme
