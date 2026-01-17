// Copyright (c) 2021-2024. LLNL-CODE-2000766 and Mneme Contributors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm-c/BitReader.h"
#include <iostream>
#include <llvm/IR/Module.h>
#include <mneme/MnemeLogger.hpp>

#include "core.h"

extern "C" {

API_EXPORT(LLVMModuleRef)
LLVMPY_ParseBitcode(LLVMContextRef context, const char *bitcode,
                    size_t bitcodelen, char **outmsg) {
  LLVMModuleRef ref;
  LLVMMemoryBufferRef mem = LLVMCreateMemoryBufferWithMemoryRange(
      bitcode, bitcodelen, "" /* BufferName*/, 0 /* RequiresNullTerminator*/
  );

  LLVMParseBitcodeInContext(context, mem, &ref, outmsg);
  LLVMDisposeMemoryBuffer(mem);
  return ref;
}

} // end extern "C"
