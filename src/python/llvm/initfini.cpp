// Copyright (c) 2021-2024. LLNL-CODE-2000766 and Mneme Contributors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "core.h"
#include "llvm/Config/llvm-config.h"
#include <iostream>
#include <proteus/CoreLLVM.h>

namespace {
class LLVMInstance {
public:
  proteus::InitLLVMTargets Init;
  static LLVMInstance &getInstance() {
    static LLVMInstance instance;
    return instance;
  }

  // Delete copy constructor and assignment operator
  LLVMInstance(const LLVMInstance &) = delete;
  LLVMInstance &operator=(const LLVMInstance &) = delete;

private:
  LLVMInstance() {}
  ~LLVMInstance() {}
};

}; // namespace

extern "C" {

API_EXPORT(void) LLVMPY_Initialize() {
  LLVMInstance &Instance = LLVMInstance::getInstance();
}

API_EXPORT(unsigned int)
LLVMPY_GetVersionInfo() {
  unsigned int verinfo = 0;
  verinfo += LLVM_VERSION_MAJOR << 16;
  verinfo += LLVM_VERSION_MINOR << 8;
#ifdef LLVM_VERSION_PATCH
  /* Not available under Windows... */
  verinfo += LLVM_VERSION_PATCH << 0;
#endif
  return verinfo;
}

} // end extern "C"
