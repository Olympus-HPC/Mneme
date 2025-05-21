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
