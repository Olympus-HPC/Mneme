#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <vector>
#include <sys/types.h>
#include <utility>

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeAnnotation.hpp"
#include "mneme/MnemeAnnotationInternal.hpp"
#include "mneme/MnemeComparators.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeUtils.hpp"

namespace mneme {
// Metadata attached to a byte subrange inside one owning blob.
struct MemoryRegionMetadata {
  uint64_t Offset = 0;
  uint64_t Extent = 0;
  Metadata MD;

  uint64_t endOffset() const { return Offset + Extent; }

  bool contains(uint64_t ByteOffset) const {
    return Offset <= ByteOffset && ByteOffset < endOffset();
  }

  bool hasSameRange(uint64_t OtherOffset, uint64_t OtherExtent) const {
    return Offset == OtherOffset && Extent == OtherExtent;
  }

  bool overlaps(uint64_t OtherOffset, uint64_t OtherExtent) const {
    return Offset < OtherOffset + OtherExtent &&
           OtherOffset < Offset + Extent;
  }
};

inline bool operator==(const MemoryRegionMetadata &LHS,
                       const MemoryRegionMetadata &RHS) {
  return LHS.Offset == RHS.Offset && LHS.Extent == RHS.Extent &&
         LHS.MD == RHS.MD;
}

inline bool operator!=(const MemoryRegionMetadata &LHS,
                       const MemoryRegionMetadata &RHS) {
  return !(LHS == RHS);
}

template <DeviceVendors VendorTypes> class MnemeMemoryBlob {
public:
  using MnemeDeviceRT = DeviceTraits<VendorTypes>;
  using DeviceError_t = typename MnemeDeviceRT::DeviceError_t;
  using DeviceStream_t = typename MnemeDeviceRT::DeviceStream_t;
  using KernelFunction_t = typename MnemeDeviceRT::KernelFunction_t;
  using MemoryAllocationHandle_t =
      typename MnemeDeviceRT::MemoryAllocationHandle_t;

protected:
  static bool compareRange(const char *Expected, const char *Actual,
                           uint64_t NumBytes, const Metadata &Md) {
    if (NumBytes == 0)
      return true;

    auto Compare = compareDeviceBlobs(Expected, Actual, NumBytes, Md);

    // Comparator semantics:
    // - Norm::None  => per-element thresholding reports AnyFail/FirstBadIdx
    // - Norm::L1/L2/Linf => aggregated error is reported in Agg
    if (Md.norm == Norm::None)
      return Compare.AnyFail == 0;

    return Compare.Agg <= Md.threshold;
  }

  Metadata PtrMD;
  std::vector<MemoryRegionMetadata> RegionMD;
  uint64_t ActualSize;
  void *BlobAddr;
  uint64_t Size;
  std::unique_ptr<uint8_t[]> HostData;
  bool IsMapped;

  bool isValidRegionRange(uint64_t Offset, uint64_t Extent) const {
    return Extent != 0 && Offset <= Size && Extent <= Size - Offset;
  }

  static bool regionSortLess(const MemoryRegionMetadata &LHS,
                             const MemoryRegionMetadata &RHS) {
    if (LHS.Offset != RHS.Offset)
      return LHS.Offset < RHS.Offset;
    return LHS.Extent < RHS.Extent;
  }

public:
  MnemeMemoryBlob(uint64_t ActualSize = 0, void *BlobAddr = nullptr,
                  uint64_t Size = 0)
      : ActualSize(ActualSize), BlobAddr(BlobAddr), Size(Size),
        HostData(new uint8_t[Size]), IsMapped(false) {}

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

  static std::pair<void *, MnemeMemoryBlob<VendorTypes>>
  fromBuffer(const char *&Buffer) {
    size_t ActualSize = util::extractScalar<size_t>(Buffer);
    size_t Size = util::extractScalar<size_t>(Buffer);
    void *DeviceAddr = util::extractScalar<void *>(Buffer);
    auto Blob = MnemeMemoryBlob<VendorTypes>(ActualSize, 0, Size);
    std::memcpy(Blob.getHostData().get(), Buffer, Size);
    Buffer += Size;
    LOG_DEBUG("Read memory blob at address {} SIZE: {} ActualSize:{}",
              DeviceAddr, Size, ActualSize);
    Blob.PtrMD = metadata::fromBuffer(Buffer);
    return std::make_pair(DeviceAddr, std::move(Blob));
  }

  void *ptr() { return reinterpret_cast<void *>(BlobAddr); }

  MnemeMemoryBlob(const MnemeMemoryBlob &) = delete;
  MnemeMemoryBlob &operator=(const MnemeMemoryBlob &) = delete;

  MnemeMemoryBlob &operator=(MnemeMemoryBlob &&other) noexcept {
    if (this != &other) {
      BlobAddr = other.BlobAddr;
      Size = other.Size;
      ActualSize = other.ActualSize;
      HostData = std::move(other.HostData);
      IsMapped = other.IsMapped;
      PtrMD = other.PtrMD;
      RegionMD = std::move(other.RegionMD);
      other.BlobAddr = 0;
      other.HostData = nullptr;
    }
    return *this;
  }

  MnemeMemoryBlob(MnemeMemoryBlob &&other) noexcept
      : BlobAddr(other.BlobAddr), Size(other.Size),
        ActualSize(other.ActualSize), HostData(std::move(other.HostData)),
        IsMapped(other.IsMapped), PtrMD(other.PtrMD),
        RegionMD(std::move(other.RegionMD)) {
    other.BlobAddr = 0;
    other.HostData = nullptr;
  }

  template <DeviceVendors VendorTypes_>
  friend llvm::raw_ostream &
  operator<<(llvm::raw_ostream &OS, const MnemeMemoryBlob<VendorTypes_> &Blob);

  void *getBlobAddr() const { return BlobAddr; }
  uint64_t getActualSize() const { return ActualSize; }
  uint64_t getSize() const { return Size; }
  const std::unique_ptr<uint8_t[]> &getHostData() const { return HostData; }

  void setMetadata(Metadata Md) { PtrMD = Md; }

  Metadata getMetadata() const { return PtrMD; }

  // Distinct overlapping regions are rejected. Re-registering the exact same
  // [offset, extent) updates the existing metadata in place.
  bool setRegionMetadata(uint64_t Offset, uint64_t Extent, Metadata Md) {
    if (!isValidRegionRange(Offset, Extent))
      return false;

    MemoryRegionMetadata Key{Offset, Extent, {}};
    auto It = std::lower_bound(RegionMD.begin(), RegionMD.end(), Key,
                               regionSortLess);

    if (It != RegionMD.end() && It->hasSameRange(Offset, Extent)) {
      It->MD = std::move(Md);
      return true;
    }

    if (It != RegionMD.begin()) {
      auto Prev = std::prev(It);
      if (Prev->overlaps(Offset, Extent))
        return false;
    }
    if (It != RegionMD.end() && It->overlaps(Offset, Extent))
      return false;

    RegionMD.insert(It, {Offset, Extent, std::move(Md)});
    return true;
  }

  // Replace the entire region table after validating a bulk-loaded set, such
  // as snapshot deserialization.
  bool replaceRegionMetadata(std::vector<MemoryRegionMetadata> Regions) {
    for (const auto &Region : Regions) {
      if (!isValidRegionRange(Region.Offset, Region.Extent))
        return false;
    }

    std::sort(Regions.begin(), Regions.end(), regionSortLess);
    for (size_t I = 1; I < Regions.size(); ++I) {
      if (Regions[I - 1].overlaps(Regions[I].Offset, Regions[I].Extent))
        return false;
    }

    RegionMD = std::move(Regions);
    return true;
  }

  void clearRegionMetadata() { RegionMD.clear(); }

  bool hasRegionMetadata() const { return !RegionMD.empty(); }

  const std::vector<MemoryRegionMetadata> &getRegionMetadata() const {
    return RegionMD;
  }

  // RegionMD is kept sorted by offset, so the owning region for a byte can be
  // resolved by finding the last region whose start is <= ByteOffset.
  const MemoryRegionMetadata *findRegionMetadata(uint64_t ByteOffset) const {
    auto It = std::upper_bound(
        RegionMD.begin(), RegionMD.end(), ByteOffset,
        [](uint64_t Offset, const MemoryRegionMetadata &Region) {
          return Offset < Region.Offset;
        });
    if (It == RegionMD.begin())
      return nullptr;

    --It;
    return It->contains(ByteOffset) ? &*It : nullptr;
  }

  bool operator==(const MnemeMemoryBlob<VendorTypes> &other) const {
    if (getSize() != other.getSize()) {
      LOG_WARN("Sizes Differ {} vs {}", getSize(), other.getSize());
      return false;
    }

    if (getMetadata() != other.getMetadata()) {
      LOG_WARN("Whole-blob metadata differs for blob {}", getBlobAddr());
      return false;
    }

    // Compare the region tables first so the subsequent byte comparison can
    // assume both blobs partition the address range the same way.
    const auto &Regions = getRegionMetadata();
    const auto &OtherRegions = other.getRegionMetadata();

    if (Regions.size() != OtherRegions.size()) {
      LOG_WARN("Region metadata count differs for blob {}", getBlobAddr());
      return false;
    }

    for (size_t I = 0; I < Regions.size(); ++I) {
      const auto &ExpectedRegion = Regions[I];
      const auto &ActualRegion = OtherRegions[I];
      if (ExpectedRegion != ActualRegion) {
        LOG_WARN("Region metadata differs for blob {} at region index {}",
                 getBlobAddr(), I);
        return false;
      }
    }

    if (Regions.empty())
      return compareRange(static_cast<const char *>(other.getBlobAddr()),
                          static_cast<const char *>(getBlobAddr()), getSize(),
                          getMetadata());

    uint64_t CurrentOffset = 0;
    for (const auto &Region : Regions) {
      if (!compareRange(static_cast<const char *>(other.getBlobAddr()) +
                            CurrentOffset,
                        static_cast<const char *>(getBlobAddr()) +
                            CurrentOffset,
                        Region.Offset - CurrentOffset, getMetadata())) {
        return false;
      }

      if (!compareRange(static_cast<const char *>(other.getBlobAddr()) +
                            Region.Offset,
                        static_cast<const char *>(getBlobAddr()) +
                            Region.Offset,
                        Region.Extent, Region.MD)) {
        return false;
      }

      CurrentOffset = Region.endOffset();
    }

    return compareRange(static_cast<const char *>(other.getBlobAddr()) +
                            CurrentOffset,
                        static_cast<const char *>(getBlobAddr()) +
                            CurrentOffset,
                        getSize() - CurrentOffset, getMetadata());
  }
  bool operator!=(const MnemeMemoryBlob<VendorTypes> &other) const {
    return !(*this == other);
  }
};

template <DeviceVendors VendorTypes>
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const MnemeMemoryBlob<VendorTypes> &Blob) {
  // The format in the binary is the following:
  // | Var Actual Size | Var-Size | Device Address | Var Data | Metadata
  OS << llvm::StringRef(reinterpret_cast<const char *>(&Blob.ActualSize),
                        sizeof(Blob.ActualSize));
  OS << llvm::StringRef(reinterpret_cast<const char *>(&Blob.Size),
                        sizeof(Blob.Size));
  OS << llvm::StringRef(reinterpret_cast<const char *>(&Blob.BlobAddr),
                        sizeof(Blob.BlobAddr));

  LOG_DEBUG("Serializing MemoryBlob, DevAddr:{} MirroredHostAddr:{} Size:{} "
            "ActualSize:{}",
            (void *)Blob.BlobAddr,
            reinterpret_cast<void *>(Blob.getHostData().get()), Blob.getSize(),
            Blob.getActualSize());

  auto EC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
      DeviceTraits<VendorTypes>::DeviceCopy(
          static_cast<void *>(Blob.getHostData().get()),
          reinterpret_cast<void *>(Blob.BlobAddr), Blob.getSize(),
          DeviceTraits<VendorTypes>::MemcpyDeviceToHostKind()));
  if (EC)
    LOG_FATAL("Error in copying data from device when serializing context on "
              "output stream\nDevice Error Msg: " +
              EC.value() + "\n");
  OS << llvm::StringRef(
      reinterpret_cast<const char *>(Blob.getHostData().get()), Blob.Size);
  auto MD = Blob.getMetadata();
  mneme::metadata::serialize(OS, MD);

  return OS;
}
} // namespace mneme
