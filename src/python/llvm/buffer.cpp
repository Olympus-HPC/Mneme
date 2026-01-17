// Copyright (c) 2021-2024. LLNL-CODE-2000766 and Mneme Contributors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "core.h"
#include "llvm-c/Types.h"
#include <llvm/Support/CBindingWrapping.h>
#include <llvm/Support/MemoryBuffer.h>

extern "C" {
API_EXPORT(void)
LLVMPY_DisposeMemBuffer(LLVMMemoryBufferRef Buffer) {
  llvm::MemoryBuffer *buff = llvm::unwrap(Buffer);
  delete buff;
}

API_EXPORT(size_t) LLVMPY_GetMemBufferSize(LLVMMemoryBufferRef Buffer) {
  auto *buff = llvm::unwrap(Buffer);
  return buff->getBufferSize();
}
}
