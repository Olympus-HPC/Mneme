#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <sys/types.h>
#include <utility>

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeAnnotation.hpp"
#include "mneme/MnemeAnnotationInternal.hpp"
#include "mneme/MnemeComparators.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeSnapshotRecords.hpp"
#include "mneme/MnemeUtils.hpp"

namespace mneme {
template <DeviceVendors VendorTypes> class MnemeMemoryBlob {
public:
  using MnemeDeviceRT = DeviceTraits<VendorTypes>;
  using DeviceError_t = typename MnemeDeviceRT::DeviceError_t;
  using DeviceStream_t = typename MnemeDeviceRT::DeviceStream_t;
  using KernelFunction_t = typename MnemeDeviceRT::KernelFunction_t;
  using MemoryAllocationHandle_t =
      typename MnemeDeviceRT::MemoryAllocationHandle_t;

protected:
  Metadata PtrMD;
  // Identifies the blob across snapshots independently of its device address.
  uint64_t BlobId;
  // Offset from the start of the recorded VA reservation.
  uint64_t BlobOffset;
  uint64_t ActualSize;
  void *BlobAddr;
  uint64_t Size;
  std::unique_ptr<uint8_t[]> HostData;
  bool IsMapped;

public:
  MnemeMemoryBlob(uint64_t ActualSize = 0, void *BlobAddr = nullptr,
                  uint64_t Size = 0, uint64_t BlobId = 0,
                  uint64_t BlobOffset = 0)
      : PtrMD(), BlobId(BlobId), BlobOffset(BlobOffset), ActualSize(ActualSize),
        BlobAddr(BlobAddr), Size(Size), HostData(new uint8_t[Size]),
        IsMapped(false) {}

  DeviceError_t map(void *VA, uint64_t ActualSize, uint64_t Size) {
    this->Size = Size;
    // We need to pass here "ActualSize". As device allocators depend on page
    // aligned allocations
    this->BlobAddr = VA;
    this->IsMapped = true;
    this->ActualSize = ActualSize;
    return MnemeDeviceRT::DeviceSuccess;
  };

  DeviceError_t allocate(size_t Size) {
    int _device;
    auto ret = MnemeDeviceRT::DeviceMalloc(&(this->BlobAddr), Size);
    MnemeDeviceRT::getDevice(_device);
    LOG_DEBUG("Loading epilogue for device {}", _device);
    this->ActualSize = Size;
    this->Size = Size;
    this->IsMapped = false;
    return ret;
  }

  DeviceError_t release() {
    if (!BlobAddr)
      return MnemeDeviceRT::DeviceSuccess;

    if (!IsMapped) {
      auto ret = MnemeDeviceRT::DeviceFree(BlobAddr);
      BlobAddr = 0;
      return ret;
    }
    BlobAddr = 0;
    return MnemeDeviceRT::DeviceSuccess;
  }

  void setHostData(std::unique_ptr<uint8_t[]> HostData) {
    this->HostData = std::move(HostData);
  }

  ~MnemeMemoryBlob() {
    if (BlobAddr != 0 && IsMapped) {
      LOG_FATAL("Destroying memory descriptor without releasing device memory at addr={}, size={}",
                (void*)BlobAddr, Size);
    }
  }

  void copyFromDevice() const {
    auto EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceCopy(
        static_cast<void *>(HostData.get()), BlobAddr, Size,
        MnemeDeviceRT::MemcpyDeviceToHostKind()));
    if (EC)
      LOG_FATAL("Error copying blob data from device\nDevice Error Msg: " +
                EC.value() + "\n");
  }

  void *ptr() { return reinterpret_cast<void *>(BlobAddr); }

  MnemeMemoryBlob(const MnemeMemoryBlob &) = delete;
  MnemeMemoryBlob &operator=(const MnemeMemoryBlob &) = delete;

  MnemeMemoryBlob &operator=(MnemeMemoryBlob &&other) noexcept {
    if (this != &other) {
      BlobAddr = other.BlobAddr;
      Size = other.Size;
      ActualSize = other.ActualSize;
      BlobId = other.BlobId;
      BlobOffset = other.BlobOffset;
      HostData = std::move(other.HostData);
      IsMapped = other.IsMapped;
      PtrMD = other.PtrMD;
      other.BlobAddr = 0;
      other.HostData = nullptr;
    }
    return *this;
  }

  MnemeMemoryBlob(MnemeMemoryBlob &&other) noexcept
      : PtrMD(other.PtrMD), BlobId(other.BlobId), BlobOffset(other.BlobOffset),
        ActualSize(other.ActualSize), BlobAddr(other.BlobAddr),
        Size(other.Size), HostData(std::move(other.HostData)),
        IsMapped(other.IsMapped) {
    other.BlobAddr = 0;
    other.HostData = nullptr;
  }

  void *getBlobAddr() const { return BlobAddr; }
  uint64_t getBlobId() const { return BlobId; }
  uint64_t getBlobOffset() const { return BlobOffset; }
  uint64_t getActualSize() const { return ActualSize; }
  uint64_t getSize() const { return Size; }
  const std::unique_ptr<uint8_t[]> &getHostData() const { return HostData; }
  void setBlobId(uint64_t NewBlobId) { BlobId = NewBlobId; }
  void setBlobOffset(uint64_t NewBlobOffset) { BlobOffset = NewBlobOffset; }

  void setMetadata(Metadata Md) { PtrMD = Md; }

  Metadata getMetadata() const { return PtrMD; }

  bool operator==(const MnemeMemoryBlob<VendorTypes> &other) const {
    if (getSize() != other.getSize()) {
      LOG_WARN("Sizes Differ {} vs {}", getSize(), other.getSize());
      return false;
    }

    auto Md = getMetadata();
    auto Compare = compareDeviceBlobs((const char *)other.getBlobAddr(),
                                      (const char *)getBlobAddr(), getSize(),
                                      Md);

    // Comparator semantics:
    // - Norm::None  => per-element thresholding reports AnyFail/FirstBadIdx
    // - Norm::L1/L2/Linf => aggregated error is reported in Agg
    if (Md.norm == Norm::None)
      return Compare.AnyFail == 0;

    return Compare.Agg <= Md.threshold;
  }
  bool operator!=(const MnemeMemoryBlob<VendorTypes> &other) const {
    return !(*this == other);
  }
};

} // namespace mneme
