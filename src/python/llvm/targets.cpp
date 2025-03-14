#include "core.h"
#include "llvm-c/TargetMachine.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"

#include <cstdio>
#include <cstring>
#include <sstream>

namespace llvm {

inline Target *unwrap(LLVMTargetRef T) { return reinterpret_cast<Target *>(T); }

inline TargetMachine *unwrap(LLVMTargetMachineRef TM) {
  return reinterpret_cast<TargetMachine *>(TM);
}

inline LLVMTargetMachineRef wrap(TargetMachine *TM) {
  return reinterpret_cast<LLVMTargetMachineRef>(TM);
}

} // namespace llvm

extern "C" {

API_EXPORT(int)
LLVMPY_GetTripleObjectFormat(const char *tripleStr) {
  return llvm::Triple(tripleStr).getObjectFormat();
}

API_EXPORT(void)
LLVMPY_CopyStringRepOfTargetData(LLVMTargetDataRef TD, char **Out) {
  *Out = LLVMCopyStringRepOfTargetData(TD);
}

API_EXPORT(void)
LLVMPY_DisposeTargetData(LLVMTargetDataRef TD) { LLVMDisposeTargetData(TD); }

API_EXPORT(long long)
LLVMPY_ABISizeOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty) {
  return (long long)LLVMABISizeOfType(TD, Ty);
}

API_EXPORT(long long)
LLVMPY_OffsetOfElement(LLVMTargetDataRef TD, LLVMTypeRef Ty, int Element) {
  llvm::Type *tp = llvm::unwrap(Ty);
  if (!tp->isStructTy())
    return -1;
  return (long long)LLVMOffsetOfElement(TD, Ty, Element);
}

API_EXPORT(long long)
LLVMPY_ABISizeOfElementType(LLVMTargetDataRef TD, LLVMTypeRef Ty) {
  llvm::Type *tp = llvm::unwrap(Ty);
  return -1;
  //    if (!tp->isPointerTy())
  //        return -1;
  //    tp = tp->getPointerElementType();
  //    return (long long)LLVMABISizeOfType(TD, llvm::wrap(tp));
}

API_EXPORT(long long)
LLVMPY_ABIAlignmentOfElementType(LLVMTargetDataRef TD, LLVMTypeRef Ty) {
  llvm::Type *tp = llvm::unwrap(Ty);
  return -1;
  //    if (!tp->isPointerTy())
  //        return -1;
  //    tp = tp->getPointerElementType();
  //    return (long long)LLVMABIAlignmentOfType(TD, llvm::wrap(tp));
}

API_EXPORT(LLVMTargetRef)
LLVMPY_GetTargetFromTriple(const char *Triple, const char **ErrOut) {
  char *ErrorMessage;
  LLVMTargetRef T;
  if (LLVMGetTargetFromTriple(Triple, &T, &ErrorMessage)) {
    *ErrOut = LLVMPY_CreateString(ErrorMessage);
    LLVMDisposeMessage(ErrorMessage);
    return NULL;
  }
  return T;
}

API_EXPORT(const char *)
LLVMPY_GetTargetName(LLVMTargetRef T) { return LLVMGetTargetName(T); }

API_EXPORT(const char *)
LLVMPY_GetTargetDescription(LLVMTargetRef T) {
  return LLVMGetTargetDescription(T);
}

API_EXPORT(void)
LLVMPY_DisposeTargetMachine(LLVMTargetMachineRef TM) {
  return LLVMDisposeTargetMachine(TM);
}

API_EXPORT(void)
LLVMPY_GetTargetMachineTriple(LLVMTargetMachineRef TM, const char **Out) {
  // result is already strdup()ed by LLVMGetTargetMachineTriple
  *Out = LLVMGetTargetMachineTriple(TM);
}

API_EXPORT(void)
LLVMPY_SetTargetMachineAsmVerbosity(LLVMTargetMachineRef TM, int verbose) {
  LLVMSetTargetMachineAsmVerbosity(TM, verbose);
}

API_EXPORT(const void *)
LLVMPY_GetBufferStart(LLVMMemoryBufferRef MB) { return LLVMGetBufferStart(MB); }

API_EXPORT(size_t)
LLVMPY_GetBufferSize(LLVMMemoryBufferRef MB) { return LLVMGetBufferSize(MB); }

API_EXPORT(void)
LLVMPY_DisposeMemoryBuffer(LLVMMemoryBufferRef MB) {
  return LLVMDisposeMemoryBuffer(MB);
}

} // end extern "C"
