#pragma once
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeUtils.hpp"
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


  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const GlobalVarInfo &GVar);
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       GlobalVarInfo &GVar);
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, GlobalVarInfo &GVar);
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const GlobalVarInfo &GVar);

} // namespace mneme
