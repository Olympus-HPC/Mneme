#pragma once
#include "mneme/MnemeLogger.hpp"
#include "mneme/Utils.hpp"
#include <cstdint>
#include <cstring>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StableHashing.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>

namespace mneme {
struct KernelInfo {
  void **Handle;
  std::string Name;
  uint64_t StaticHash;
  llvm::SmallVector<size_t> KernelArgSizes;
  llvm::SmallVector<std::string> KernelArgNames;
  llvm::SmallVector<std::function<double(void *)>> ToDoubleFunc;
  llvm::SmallVector<bool> KernelSpecializations;
  llvm::SmallVector<std::string> ModuleFiles;
  llvm::SmallVector<std::unique_ptr<uint8_t[]>> ArgData;
  KernelInfo(void **Handle, char *Name)
      : Handle(Handle), Name(Name),
        StaticHash(llvm::stable_hash_combine_string(Name)) {};
  KernelInfo(void **Handle, std::string &Name)
      : Handle(Handle), Name(Name),
        StaticHash(llvm::stable_hash_combine_string(Name)) {};

  KernelInfo() : Name(nullptr) {};

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
      FATAL_ERROR("Setting argument data out of range");

    auto MemSize = KernelArgSizes[Index];
    ArgData[Index] = std::make_unique<uint8_t[]>(MemSize);
    std::memcpy(static_cast<void *>(ArgData[Index].get()),
                static_cast<const void *>(Data), MemSize);
    Data += KernelArgSizes[Index];
  }

  void setToDoubleFunc(
      llvm::ArrayRef<std::function<double(void *)>> KernelArgNames) {
    ToDoubleFunc =
        llvm::SmallVector<std::function<double(void *)>>(KernelArgNames);
  }

  void updateHash(uint64_t Hash) {
    StaticHash = llvm::stable_hash_combine(StaticHash, Hash);
  }

  int64_t getNumArgs() const { return KernelArgSizes.size(); }
  void **getHandle() const { return Handle; }
  const std::string getName() const { return Name; }
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

struct GlobalVarInfo {
  std::string Name;
  const void *HostSymbolAddr;
  void *DevAddr;
  std::shared_ptr<uint8_t[]> HostAddr;
  size_t VarSize;
  GlobalVarInfo(const char *Name, const void *HostSymbolAddr, size_t VarSize,
                void *DevAddr = nullptr)
      : Name(Name), HostSymbolAddr(HostSymbolAddr), VarSize(VarSize),
        DevAddr(DevAddr), HostAddr(std::make_unique<uint8_t[]>(VarSize)) {};
  GlobalVarInfo(std::string &Name, const void *HostSymbolAddr, size_t VarSize,
                void *DevAddr = nullptr)
      : Name(Name), HostSymbolAddr(HostSymbolAddr), VarSize(VarSize),
        DevAddr(DevAddr), HostAddr(std::make_unique<uint8_t[]>(VarSize)) {};

  ~GlobalVarInfo() {}

  GlobalVarInfo(GlobalVarInfo &&other) noexcept = default;
  GlobalVarInfo &operator=(GlobalVarInfo &&other) noexcept = default;

  GlobalVarInfo &operator=(const GlobalVarInfo &other) {
    if (this != &other) {
      Name = other.Name;
      HostSymbolAddr = other.HostSymbolAddr;
      DevAddr = other.DevAddr;
      HostAddr = other.HostAddr;
      VarSize = other.VarSize;
    }
    return *this;
  }

  GlobalVarInfo(const GlobalVarInfo &other) noexcept
      : Name(other.Name), HostSymbolAddr(other.HostSymbolAddr),
        DevAddr(other.DevAddr), VarSize(other.VarSize),
        HostAddr(other.HostAddr) {}

  static GlobalVarInfo fromBuffer(const char *&Buffer) {
    const char *tmp = Buffer;
    size_t StrLen = util::extractScalar<size_t>(Buffer);
    std::string Name{Buffer, StrLen};
    Buffer += StrLen;
    size_t VarSize = util::extractScalar<size_t>(Buffer);
    void *DevAddr = util::extractScalar<void *>(Buffer);
    GlobalVarInfo GV(Name, nullptr, VarSize, DevAddr);
    std::memcpy(GV.HostAddr.get(), Buffer, VarSize);
    Buffer += VarSize;
    LOG_DEBUG("Loaded from buffer Global, Name:{}, VarSize:{}, RecoredAddr:{}",
              Name, VarSize, DevAddr);
    return std::move(GV);
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const GlobalVarInfo &GVar);
};

// Overload operator<<
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const GlobalVarInfo &GVar) {
  // The format in the binary is the following:
  // | Var-Name-Size | Var-Name | Var-Size | Device Address | Var Data |
  size_t StrLen = GVar.Name.size();
  OS << llvm::StringRef(reinterpret_cast<const char *>(&StrLen),
                        sizeof(StrLen));
  OS << GVar.Name;
  OS << llvm::StringRef(reinterpret_cast<const char *>(&GVar.VarSize),
                        sizeof(GVar.VarSize));
  OS << llvm::StringRef(reinterpret_cast<const char *>(&GVar.DevAddr),
                        sizeof(GVar.DevAddr));
  OS << llvm::StringRef(reinterpret_cast<const char *>(GVar.HostAddr.get()),
                        GVar.VarSize);

  return OS;
}
} // namespace mneme
