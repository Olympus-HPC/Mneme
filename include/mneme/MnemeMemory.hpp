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
#include "mneme/MnemeLLVMUtils.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeUtils.hpp"

namespace mneme {
struct ByteSpan {
  uint64_t Offset = 0;
  uint64_t Extent = 0;

  uint64_t endOffset() const { return Offset + Extent; }

  bool contains(uint64_t ByteOffset) const {
    return Offset <= ByteOffset && ByteOffset < endOffset();
  }

  bool hasSameRange(uint64_t OtherOffset, uint64_t OtherExtent) const {
    return Offset == OtherOffset && Extent == OtherExtent;
  }

  bool hasSameRange(const ByteSpan &Other) const {
    return hasSameRange(Other.Offset, Other.Extent);
  }

  bool overlaps(uint64_t OtherOffset, uint64_t OtherExtent) const {
    return Offset < OtherOffset + OtherExtent && OtherOffset < endOffset();
  }

  bool overlaps(const ByteSpan &Other) const {
    return overlaps(Other.Offset, Other.Extent);
  }
};

inline bool operator==(const ByteSpan &LHS, const ByteSpan &RHS) {
  return LHS.Offset == RHS.Offset && LHS.Extent == RHS.Extent;
}

inline bool operator!=(const ByteSpan &LHS, const ByteSpan &RHS) {
  return !(LHS == RHS);
}

// Metadata attached to a byte range inside one owning blob.
struct MemoryAnnotation {
  ByteSpan Range;
  Metadata MD;
};

inline bool operator==(const MemoryAnnotation &LHS, const MemoryAnnotation &RHS) {
  return LHS.Range == RHS.Range && LHS.MD == RHS.MD;
}

inline bool operator!=(const MemoryAnnotation &LHS, const MemoryAnnotation &RHS) {
  return !(LHS == RHS);
}

inline std::vector<MemoryAnnotation>
readAnnotationsFromBuffer(const char *&Buffer, size_t NumAnnotations) {
  std::vector<MemoryAnnotation> LoadedAnnotations;
  LoadedAnnotations.reserve(NumAnnotations);
  for (size_t I = 0; I < NumAnnotations; ++I) {
    MemoryAnnotation Annotation;
    Annotation.Range.Offset = util::extractScalar<uint64_t>(Buffer);
    Annotation.Range.Extent = util::extractScalar<uint64_t>(Buffer);
    Annotation.MD = metadata::fromBuffer(Buffer);
    LoadedAnnotations.push_back(std::move(Annotation));
  }
  return LoadedAnnotations;
}

inline void writeAnnotationsToStream(
    llvm::raw_ostream &OS, const std::vector<MemoryAnnotation> &Annotations) {
  util::writeScalar(OS, Annotations.size());
  for (const auto &Annotation : Annotations) {
    util::writeScalar(OS, Annotation.Range.Offset);
    util::writeScalar(OS, Annotation.Range.Extent);
    metadata::serialize(OS, Annotation.MD);
  }
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

  std::vector<MemoryAnnotation> Annotations;
  uint64_t ActualSize;
  void *BlobAddr;
  uint64_t Size;
  std::unique_ptr<uint8_t[]> HostData;
  bool IsMapped;

  bool isValidAnnotationRange(const ByteSpan &Range) const {
    if (Range.Offset > Size)
      return false;
    if (Range.Extent == 0)
      return Size == 0 && Range.Offset == 0;
    return Range.Extent <= Size - Range.Offset;
  }

  bool isWholeBlobRange(const ByteSpan &Range) const {
    return Range.Offset == 0 && Range.Extent == Size;
  }

  static bool annotationSortLess(const MemoryAnnotation &LHS,
                                 const MemoryAnnotation &RHS) {
    if (LHS.Range.Offset != RHS.Range.Offset)
      return LHS.Range.Offset < RHS.Range.Offset;
    return LHS.Range.Extent < RHS.Range.Extent;
  }

  void resetDefaultAnnotation() { Annotations = {{{0, Size}, {}}}; }

  const MemoryAnnotation *findWholeBlobAnnotation() const {
    for (const auto &Annotation : Annotations) {
      if (isWholeBlobRange(Annotation.Range))
        return &Annotation;
    }
    return nullptr;
  }

  static std::vector<MemoryAnnotation>
  regionAnnotations(const std::vector<MemoryAnnotation> &Annotations,
                    uint64_t BlobSize) {
    std::vector<MemoryAnnotation> Regions;
    Regions.reserve(Annotations.size());
    for (const auto &Annotation : Annotations) {
      if (Annotation.Range.Offset == 0 && Annotation.Range.Extent == BlobSize)
        continue;
      Regions.push_back(Annotation);
    }
    return Regions;
  }

public:
  bool registerAnnotation(ByteSpan Range, Metadata Md) {
    if (!isValidAnnotationRange(Range))
      return false;

    for (auto &Annotation : Annotations) {
      if (Annotation.Range.hasSameRange(Range)) {
        Annotation.MD = std::move(Md);
        return true;
      }

      if (!Annotation.Range.overlaps(Range))
        continue;

      // The full-range annotation is the blob's default policy, so narrower
      // region annotations may overlap it. Overlaps between two region
      // annotations remain invalid.
      if (isWholeBlobRange(Annotation.Range) || isWholeBlobRange(Range))
        continue;

      return false;
    }

    Annotations.push_back({Range, std::move(Md)});
    std::sort(Annotations.begin(), Annotations.end(), annotationSortLess);
    return true;
  }
  MnemeMemoryBlob(uint64_t ActualSize = 0, void *BlobAddr = nullptr,
                  uint64_t Size = 0)
      : ActualSize(ActualSize), BlobAddr(BlobAddr), Size(Size),
        HostData(new uint8_t[Size]), IsMapped(false) {
    resetDefaultAnnotation();
  }

  DeviceError_t map(void *VA, uint64_t ActualSize, uint64_t Size) {
    this->Size = Size;
    // We need to pass here "ActualSize". As device allocators depend on page
    // aligned allocations
    this->BlobAddr = VA;
    this->IsMapped = true;
    this->ActualSize = ActualSize;
    // Preserve any annotations already loaded from a snapshot. Fresh blobs get
    // their default whole-blob annotation from the constructor.
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
    // Preserve any annotations already loaded from a snapshot. Fresh blobs get
    // their default whole-blob annotation from the constructor.
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
    size_t NumAnnotations = util::extractScalar<size_t>(Buffer);
    auto LoadedAnnotations = readAnnotationsFromBuffer(Buffer, NumAnnotations);
    if (!Blob.replaceAnnotations(std::move(LoadedAnnotations)))
      LOG_FATAL("Invalid annotation set serialized for blob");
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
      Annotations = std::move(other.Annotations);
      other.BlobAddr = 0;
      other.HostData = nullptr;
    }
    return *this;
  }

  MnemeMemoryBlob(MnemeMemoryBlob &&other) noexcept
      : BlobAddr(other.BlobAddr), Size(other.Size),
        ActualSize(other.ActualSize), HostData(std::move(other.HostData)),
        IsMapped(other.IsMapped), Annotations(std::move(other.Annotations)) {
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

  bool replaceAnnotations(std::vector<MemoryAnnotation> CandidateAnnotations) {
    if (CandidateAnnotations.empty())
      return false;

    // Install a complete blob annotation set, typically after deserializing a
    // snapshot. The set must contain exactly one full-range default annotation.
    size_t WholeBlobCount = 0;
    for (const auto &Annotation : CandidateAnnotations) {
      if (!isValidAnnotationRange(Annotation.Range))
        return false;
      if (isWholeBlobRange(Annotation.Range))
        ++WholeBlobCount;
    }
    if (WholeBlobCount != 1)
      return false;

    std::sort(CandidateAnnotations.begin(), CandidateAnnotations.end(),
              annotationSortLess);
    for (size_t I = 1; I < CandidateAnnotations.size(); ++I) {
      if (CandidateAnnotations[I - 1].Range.hasSameRange(
              CandidateAnnotations[I].Range))
        continue;
      if (CandidateAnnotations[I - 1].Range.overlaps(
              CandidateAnnotations[I].Range) &&
          !isWholeBlobRange(CandidateAnnotations[I - 1].Range) &&
          !isWholeBlobRange(CandidateAnnotations[I].Range)) {
        return false;
      }
    }

    Annotations = std::move(CandidateAnnotations);
    return true;
  }

  const std::vector<MemoryAnnotation> &getAnnotations() const {
    return Annotations;
  }

  const MemoryAnnotation *getWholeBlobAnnotation() const {
    return findWholeBlobAnnotation();
  }

  std::vector<MemoryAnnotation> getRegionAnnotations() const {
    // Region annotations are the narrower ranges, excluding the
    // full-range default annotation for the blob.
    return regionAnnotations(Annotations, Size);
  }

  bool operator==(const MnemeMemoryBlob<VendorTypes> &other) const {
    if (getSize() != other.getSize()) {
      LOG_WARN("Sizes Differ {} vs {}", getSize(), other.getSize());
      return false;
    }

    auto *WholeBlob = getWholeBlobAnnotation();
    auto *OtherWholeBlob = other.getWholeBlobAnnotation();
    if (!WholeBlob || !OtherWholeBlob || WholeBlob->MD != OtherWholeBlob->MD) {
      LOG_WARN("Whole-blob metadata differs for blob {}", getBlobAddr());
      return false;
    }

    // First compare the region annotation structure, then compare bytes using
    // the whole-blob annotation as fallback for any uncovered ranges.
    auto Regions = getRegionAnnotations();
    auto OtherRegions = other.getRegionAnnotations();

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

    if (Regions.empty()) {
      LOG_DEBUG(
          "Comparing blob {} full range [0, {}) with whole-blob metadata builtin={} threshold={} threshold_kind={} norm={} tag={}",
          getBlobAddr(), getSize(),
          static_cast<unsigned>(WholeBlob->MD.builtin), WholeBlob->MD.threshold,
          static_cast<unsigned>(WholeBlob->MD.threshold_kind),
          static_cast<unsigned>(WholeBlob->MD.norm),
          WholeBlob->MD.tag.value_or("no_tag"));
      return compareRange(static_cast<const char *>(other.getBlobAddr()),
                          static_cast<const char *>(getBlobAddr()), getSize(),
                          WholeBlob->MD);
    }

    uint64_t CurrentOffset = 0;
    for (const auto &Region : Regions) {
      if (Region.Range.Offset > CurrentOffset) {
        LOG_DEBUG(
            "Comparing blob {} gap [{}, {}) with whole-blob metadata builtin={} threshold={} threshold_kind={} norm={} tag={}",
            getBlobAddr(), CurrentOffset, Region.Range.Offset,
            static_cast<unsigned>(WholeBlob->MD.builtin), WholeBlob->MD.threshold,
            static_cast<unsigned>(WholeBlob->MD.threshold_kind),
            static_cast<unsigned>(WholeBlob->MD.norm),
            WholeBlob->MD.tag.value_or("no_tag"));
      }
      if (!compareRange(static_cast<const char *>(other.getBlobAddr()) +
                            CurrentOffset,
                        static_cast<const char *>(getBlobAddr()) +
                            CurrentOffset,
                        Region.Range.Offset - CurrentOffset, WholeBlob->MD)) {
        return false;
      }

      LOG_DEBUG(
          "Comparing blob {} region [{}, {}) with region metadata builtin={} threshold={} threshold_kind={} norm={} tag={}",
          getBlobAddr(), Region.Range.Offset, Region.Range.endOffset(),
          static_cast<unsigned>(Region.MD.builtin), Region.MD.threshold,
          static_cast<unsigned>(Region.MD.threshold_kind),
          static_cast<unsigned>(Region.MD.norm),
          Region.MD.tag.value_or("no_tag"));
      if (!compareRange(static_cast<const char *>(other.getBlobAddr()) +
                            Region.Range.Offset,
                        static_cast<const char *>(getBlobAddr()) +
                            Region.Range.Offset,
                        Region.Range.Extent, Region.MD)) {
        return false;
      }

      CurrentOffset = Region.Range.endOffset();
    }

    if (CurrentOffset < getSize()) {
      LOG_DEBUG(
          "Comparing blob {} tail [{}, {}) with whole-blob metadata builtin={} threshold={} threshold_kind={} norm={} tag={}",
          getBlobAddr(), CurrentOffset, getSize(),
          static_cast<unsigned>(WholeBlob->MD.builtin), WholeBlob->MD.threshold,
          static_cast<unsigned>(WholeBlob->MD.threshold_kind),
          static_cast<unsigned>(WholeBlob->MD.norm),
          WholeBlob->MD.tag.value_or("no_tag"));
    }
    return compareRange(static_cast<const char *>(other.getBlobAddr()) +
                            CurrentOffset,
                        static_cast<const char *>(getBlobAddr()) +
                            CurrentOffset,
                        getSize() - CurrentOffset, WholeBlob->MD);
  }
  bool operator!=(const MnemeMemoryBlob<VendorTypes> &other) const {
    return !(*this == other);
  }
};

template <DeviceVendors VendorTypes>
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const MnemeMemoryBlob<VendorTypes> &Blob) {
  // The format in the binary is the following:
  // | Var Actual Size | Var-Size | Device Address | Var Data |
  // | NumAnnotations | (Offset | Extent | Metadata) * NumAnnotations
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
  writeAnnotationsToStream(OS, Blob.getAnnotations());

  return OS;
}
} // namespace mneme
