#pragma once

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StableHashing.h>
#include <llvm/Support/raw_ostream.h>
#include <cstdint>
#include <cstring>

namespace mneme {
enum class KernelArgEncodingKind : uint8_t {
  RawBytes = 0,
  ManagedPointer = 1,
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
  llvm::SmallVector<KernelArgEncodingKind> ArgEncodingKinds;
  llvm::SmallVector<uint64_t> ManagedArgBlobIds;
  llvm::SmallVector<uint64_t> ManagedArgOffsets;
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

  void initializeArgStorage(size_t NumArgs) {
    ArgData.resize(NumArgs);
    ArgEncodingKinds =
        llvm::SmallVector<KernelArgEncodingKind>(NumArgs,
                                                 KernelArgEncodingKind::RawBytes);
    ManagedArgBlobIds = llvm::SmallVector<uint64_t>(NumArgs, 0);
    ManagedArgOffsets = llvm::SmallVector<uint64_t>(NumArgs, 0);
  }

  void setRawArgData(const char *&Data, int Index) {
    if (Index >= KernelArgSizes.size() || Index >= ArgData.size())
      LOG_FATAL("Setting argument data out of range");

    auto MemSize = KernelArgSizes[Index];
    ArgData[Index] = std::make_unique<uint8_t[]>(MemSize);
    std::memcpy(static_cast<void *>(ArgData[Index].get()),
                static_cast<const void *>(Data), MemSize);
    ArgEncodingKinds[Index] = KernelArgEncodingKind::RawBytes;
    ManagedArgBlobIds[Index] = 0;
    ManagedArgOffsets[Index] = 0;
    Data += KernelArgSizes[Index];
  }

  void setManagedPointerArg(int Index, uint64_t BlobId, uint64_t Offset) {
    if (Index >= KernelArgSizes.size() || Index >= ArgData.size())
      LOG_FATAL("Setting managed pointer argument out of range");
    if (KernelArgSizes[Index] != sizeof(uintptr_t))
      LOG_FATAL("Managed pointer arg does not have pointer-sized storage");

    ArgData[Index] = std::make_unique<uint8_t[]>(sizeof(uintptr_t));
    uintptr_t Zero = 0;
    std::memcpy(static_cast<void *>(ArgData[Index].get()), &Zero, sizeof(Zero));
    ArgEncodingKinds[Index] = KernelArgEncodingKind::ManagedPointer;
    ManagedArgBlobIds[Index] = BlobId;
    ManagedArgOffsets[Index] = Offset;
  }

  void materializeManagedPointerArg(int Index, void *Ptr) {
    if (Index >= ArgData.size() || !ArgData[Index])
      LOG_FATAL("Materializing managed pointer argument out of range");
    if (ArgEncodingKinds[Index] != KernelArgEncodingKind::ManagedPointer)
      LOG_FATAL("Materializing non-managed pointer argument as managed");

    auto Value = reinterpret_cast<uintptr_t>(Ptr);
    std::memcpy(static_cast<void *>(ArgData[Index].get()), &Value,
                sizeof(Value));
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

  llvm::ArrayRef<KernelArgEncodingKind> getArgEncodingKinds() const {
    return ArgEncodingKinds;
  }

  uint64_t getManagedArgBlobId(int Index) const { return ManagedArgBlobIds[Index]; }
  uint64_t getManagedArgOffset(int Index) const { return ManagedArgOffsets[Index]; }

  llvm::ArrayRef<std::function<double(void *)>> getToDoubleFunc() const {
    return ToDoubleFunc;
  }
};
} // namespace mneme
