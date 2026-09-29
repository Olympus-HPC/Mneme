#pragma once

#include <cstdint>
#include <cstring>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StableHashing.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

namespace mneme {
struct KernelInfo {
  const void *HostFun;
  std::string Name;
  std::optional<uint64_t> StaticHash;
  llvm::SmallVector<size_t> KernelArgSizes;
  llvm::SmallVector<std::string> KernelArgNames;
  llvm::SmallVector<std::function<double(void *)>> ToDoubleFunc;
  llvm::SmallVector<bool> KernelSpecializations;
  llvm::SmallVector<std::unique_ptr<uint8_t[]>> ArgData;
  KernelInfo(const void *HostFun, const char *Name)
      : HostFun(HostFun), Name(Name), StaticHash(std::nullopt) {};
  KernelInfo(const std::string &Name) : Name(Name), StaticHash(std::nullopt) {};

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

  void initializeArgStorage(size_t NumArgs) { ArgData.resize(NumArgs); }

  // Advances Data past the copied bytes.
  void setRawArgData(const char *&Data, int Index) {
    if (Index >= KernelArgSizes.size() || Index >= ArgData.size())
      LOG_FATAL("Setting argument data out of range");

    auto MemSize = KernelArgSizes[Index];
    ArgData[Index] = std::make_unique<uint8_t[]>(MemSize);
    std::memcpy(static_cast<void *>(ArgData[Index].get()),
                static_cast<const void *>(Data), MemSize);
    Data += KernelArgSizes[Index];
  }

  void setZeroArgData(int Index) {
    if (Index >= KernelArgSizes.size() || Index >= ArgData.size())
      LOG_FATAL("Setting argument data out of range");

    ArgData[Index] = std::make_unique<uint8_t[]>(KernelArgSizes[Index]);
  }

  // Writes in place so pointers into ArgData stay valid.
  void setArgValue(int Index, const void *Value, size_t Size) {
    if (Index >= KernelArgSizes.size() || Index >= ArgData.size() ||
        !ArgData[Index])
      LOG_FATAL("Setting the value of an argument without storage");
    if (Size != KernelArgSizes[Index])
      LOG_FATAL("Argument value size " + std::to_string(Size) +
                " does not match the argument size " +
                std::to_string(KernelArgSizes[Index]));

    std::memcpy(static_cast<void *>(ArgData[Index].get()), Value, Size);
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
} // namespace mneme
