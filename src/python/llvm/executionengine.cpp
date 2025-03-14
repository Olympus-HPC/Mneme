#include "llvm-c/ExecutionEngine.h"
#include <llvm/Target/TargetMachine.h>

#include <cstdio>

namespace llvm {

inline TargetMachine *unwrap(LLVMTargetMachineRef P) {
  return reinterpret_cast<TargetMachine *>(P);
}
inline LLVMTargetMachineRef wrap(const TargetMachine *P) {
  return reinterpret_cast<LLVMTargetMachineRef>(const_cast<TargetMachine *>(P));
}
} // namespace llvm
