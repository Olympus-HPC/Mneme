#include "core.h"
#include "llvm-c/Types.h"
#include <llvm/Support/CBindingWrapping.h>
#include <llvm/Support/MemoryBuffer.h>

extern "C" {
API_EXPORT(void)
LLVMPY_DisposeTypesIter(LLVMMemoryBufferRef Buffer) {
  llvm::MemoryBuffer *buff = llvm::unwrap(Buffer);
  delete buff;
}
}
