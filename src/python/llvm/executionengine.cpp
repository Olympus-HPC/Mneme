// Copyright (c) 2021-2024. LLNL-CODE-2000766 and Mneme Contributors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
