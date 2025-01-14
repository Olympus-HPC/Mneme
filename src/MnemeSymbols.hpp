#pragma once
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StableHashing.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

namespace mneme {
struct KernelInfo {
  void **Handle;
  const char *Name;
  uint64_t StaticHash;
  llvm::SmallVector<size_t> KernelArgs;
  llvm::SmallVector<std::string> ModuleFiles;
  KernelInfo(void **Handle, char *Name)
      : Handle(Handle), Name(Name), StaticHash(0) {};
  KernelInfo() : Name(nullptr) {};

public:
  const char *getName() { return Name; }
  void setArgs(llvm::ArrayRef<size_t> ArgSizes) {
    KernelArgs = llvm::SmallVector<size_t>(ArgSizes);
  }
  void **getHandle() const { return Handle; }

  void updateHash(uint64_t Hash) {
    StaticHash = llvm::stable_hash_combine(StaticHash, Hash);
  }
};

struct GlobalVarInfo {
  const char *Name;
  const void *HostSymbolAddr;
  void *DevAddr;
  void *HostAddr;
  size_t VarSize;
  GlobalVarInfo(const char *Name, const void *HostSymbolAddr, size_t VarSize)
      : Name(Name), HostSymbolAddr(HostSymbolAddr), VarSize(VarSize),
        DevAddr(nullptr) {};

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const GlobalVarInfo &GVar);
};

// Overload operator<<
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const GlobalVarInfo &GVar) {
  // The format in the binary is the following:
  // | Var-Name-Size | Var-Name | Var-Size | Device Address | Var Data |
  auto StrLen = strlen(GVar.Name);
  OS << llvm::StringRef(reinterpret_cast<const char *>(&StrLen),
                        sizeof(StrLen));
  OS << llvm::StringRef(reinterpret_cast<const char *>(GVar.Name), StrLen);
  OS << llvm::StringRef(reinterpret_cast<const char *>(&GVar.VarSize),
                        sizeof(GVar.VarSize));
  OS << llvm::StringRef(reinterpret_cast<const char *>(&GVar.DevAddr),
                        sizeof(GVar.DevAddr));
  OS << llvm::StringRef(reinterpret_cast<const char *>(&GVar.HostAddr),
                        GVar.VarSize);

  return OS;
}
} // namespace mneme
