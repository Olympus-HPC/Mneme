#pragma once

#include <cstdint>
#include <cstring>
#include <functional>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StableHashing.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <optional>
#include <string>

#include "mneme/MnemeLogger.hpp"

namespace mneme {
struct DeviceAddressRange {
  const void *Base = nullptr;
  uint64_t Size = 0;

  bool contains(const void *Ptr) const {
    auto BaseAddr = reinterpret_cast<uintptr_t>(Base);
    auto PtrAddr = reinterpret_cast<uintptr_t>(Ptr);
    return BaseAddr && PtrAddr >= BaseAddr && PtrAddr < BaseAddr + Size;
  }

  uint64_t offsetOf(const void *Ptr) const {
    if (!contains(Ptr))
      LOG_FATAL("Pointer is outside the recorded device address range");
    return reinterpret_cast<uintptr_t>(Ptr) - reinterpret_cast<uintptr_t>(Base);
  }

  void *rebase(const void *RecordedPtr, void *ReplayBase) const {
    return reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(ReplayBase) +
                                    offsetOf(RecordedPtr));
  }
};

struct KernelArgPointerSlot {
  uint64_t ArgIndex;
  uint64_t ByteOffset;

  void validateAgainst(llvm::ArrayRef<size_t> ArgSizes) const {
    if (ArgIndex >= ArgSizes.size())
      LOG_FATAL("Recorded pointer slot references an unknown argument");
    if (ByteOffset + sizeof(void *) > ArgSizes[ArgIndex])
      LOG_FATAL("Recorded pointer slot is outside the argument storage");
  }

  void *readFrom(void **Args) const {
    const auto *ArgBytes = static_cast<const uint8_t *>(Args[ArgIndex]);
    void *RecordedPtr = nullptr;
    std::memcpy(&RecordedPtr, ArgBytes + ByteOffset, sizeof(RecordedPtr));
    return RecordedPtr;
  }

  void *readFrom(const struct KernelInfo &KInfo) const;
  void writeTo(struct KernelInfo &KInfo, void *Value) const;
};

struct KernelInfo {
  const void *HostFun;
  std::string Name;
  std::optional<uint64_t> StaticHash;
  llvm::SmallVector<size_t> KernelArgSizes;
  llvm::SmallVector<std::string> KernelArgNames;
  llvm::SmallVector<std::function<double(void *)>> ToDoubleFunc;
  llvm::SmallVector<bool> KernelSpecializations;
  llvm::SmallVector<std::unique_ptr<uint8_t[]>> ArgData;
  llvm::SmallVector<KernelArgPointerSlot> PointerSlots;
  KernelInfo(const void *HostFun, char *Name)
      : HostFun(HostFun), Name(Name), StaticHash(std::nullopt) {};
  KernelInfo(std::string &Name) : Name(Name), StaticHash(std::nullopt) {};

public:
  void setArgSizes(llvm::ArrayRef<size_t> ArgSizes) {
    KernelArgSizes = llvm::SmallVector<size_t>(ArgSizes);
  }

  void setArgNames(llvm::ArrayRef<std::string> Names) {
    KernelArgNames = llvm::SmallVector<std::string>(Names);
  }

  void setSpecializations(llvm::ArrayRef<bool> Specializations) {
    KernelSpecializations = llvm::SmallVector<bool>(Specializations);
  }

  void setArgData(const char *&Data, int Index) {
    if (Index >= KernelArgSizes.size() || Index >= ArgData.size())
      LOG_FATAL("Setting argument data out of range");

    auto MemSize = KernelArgSizes[Index];
    ArgData[Index] = std::make_unique<uint8_t[]>(MemSize);
    std::memcpy(static_cast<void *>(ArgData[Index].get()),
                static_cast<const void *>(Data), MemSize);
    Data += KernelArgSizes[Index];
  }

  void setToDoubleFunc(llvm::ArrayRef<std::function<double(void *)>> convert) {
    ToDoubleFunc = llvm::SmallVector<std::function<double(void *)>>(convert);
  }

  int64_t getNumArgs() const { return KernelArgSizes.size(); }
  const std::string getName() const { return Name; }
  const void *getFunHandle() const { return HostFun; }
  llvm::ArrayRef<size_t> getArgSizes() const { return KernelArgSizes; }
  llvm::ArrayRef<std::string> getArgNames() const { return KernelArgNames; }
  llvm::ArrayRef<bool> getArgSpecializations() const {
    return KernelSpecializations;
  }

  llvm::ArrayRef<std::unique_ptr<uint8_t[]>> getArgData() const {
    return ArgData;
  }

  llvm::ArrayRef<std::function<double(void *)>> getToDoubleFunc() const {
    return ToDoubleFunc;
  }
};

inline void *KernelArgPointerSlot::readFrom(const KernelInfo &KInfo) const {
  validateAgainst(KInfo.KernelArgSizes);
  if (ArgIndex >= KInfo.ArgData.size())
    LOG_FATAL(
        "Recorded pointer slot references argument data that was not loaded");
  const auto *Slot = KInfo.ArgData[ArgIndex].get() + ByteOffset;
  void *RecordedPtr = nullptr;
  std::memcpy(&RecordedPtr, Slot, sizeof(RecordedPtr));
  return RecordedPtr;
}

inline void KernelArgPointerSlot::writeTo(KernelInfo &KInfo,
                                          void *Value) const {
  validateAgainst(KInfo.KernelArgSizes);
  if (ArgIndex >= KInfo.ArgData.size())
    LOG_FATAL(
        "Recorded pointer slot references argument data that was not loaded");
  auto *Slot = KInfo.ArgData[ArgIndex].get() + ByteOffset;
  std::memcpy(Slot, &Value, sizeof(Value));
}
} // namespace mneme
