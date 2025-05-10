#include <mneme/MnemeSymbols.hpp>

namespace mneme {
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

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, GlobalVarInfo &GVar) {
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
