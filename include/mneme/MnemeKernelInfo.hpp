#pragma once

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>

#include "mneme/MnemeDeviceBinary.hpp"
#include "mneme/MnemeSymbols.hpp"

namespace mneme {
struct KernelInfo {
  std::optional<std::reference_wrapper<MnemeDeviceLinkedBin>> Exec;
  const void *HostFun;
  std::string Name;
  std::optional<uint64_t> StaticHash;
  llvm::SmallVector<size_t> KernelArgSizes;
  llvm::SmallVector<std::string> KernelArgNames;
  llvm::SmallVector<std::function<double(void *)>> ToDoubleFunc;
  llvm::SmallVector<bool> KernelSpecializations;
  llvm::SmallVector<std::unique_ptr<uint8_t[]>> ArgData;
  KernelInfo(MnemeDeviceLinkedBin &Executable, const void *HostFun, char *Name)
      : Exec(Executable), HostFun(HostFun), Name(Name),
        StaticHash(std::nullopt) {};
  KernelInfo(MnemeDeviceLinkedBin &Executable, std::string &Name)
      : Exec(Executable), Name(Name), StaticHash(std::nullopt) {};
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
  MnemeDeviceLinkedBin &getExecutable() const {
    if (!Exec)
      LOG_FATAL(
          "Requesting Handle on an execution with unitialized binary handle");
    return Exec->get();
  }
  const std::string getName() const { return Name; }
  const void *getFunHandle() const { return HostFun; }
  llvm::ArrayRef<size_t> getArgSizes() const { return KernelArgSizes; }
  llvm::ArrayRef<std::string> getArgNames() const { return KernelArgNames; }
  llvm::ArrayRef<bool> getArgSpecializations() const {
    return KernelSpecializations;
  }

  llvm::stable_hash getStaticHash() {
    if (StaticHash)
      return StaticHash.value();
    return computeStaticHash();
  }

  llvm::ArrayRef<std::unique_ptr<uint8_t[]>> getArgData() const {
    return ArgData;
  }

  llvm::ArrayRef<std::function<double(void *)>> getToDoubleFunc() const {
    return ToDoubleFunc;
  }

private:
  llvm::stable_hash computeStaticHash() {
    auto &Executable = getExecutable();
    StaticHash = llvm::stable_hash_combine_string(Name);
    StaticHash = llvm::stable_hash_combine(StaticHash.value(),
                                           Executable.getStaticHash());
    return StaticHash.value();
  }
};
} // namespace mneme
