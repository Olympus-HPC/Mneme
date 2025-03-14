#include "core.h"

#include "llvm-c/Support.h"

extern "C" {

API_EXPORT(const char *)
LLVMPY_CreateString(const char *msg) { return strdup(msg); }

API_EXPORT(const char *)
LLVMPY_CreateByteString(const char *buf, size_t len) {
  char *dest = (char *)malloc(len + 1);
  if (dest != NULL) {
    memcpy(dest, buf, len);
    dest[len] = '\0';
  }
  return dest;
}

API_EXPORT(void)
LLVMPY_DisposeString(const char *msg) { free(const_cast<char *>(msg)); }

API_EXPORT(LLVMContextRef)
LLVMPY_GetGlobalContext() { return LLVMGetGlobalContext(); }

API_EXPORT(LLVMContextRef)
LLVMPY_ContextCreate() { return LLVMContextCreate(); }

API_EXPORT(void)
LLVMPY_ContextDispose(LLVMContextRef context) {
  return LLVMContextDispose(context);
}

} // end extern "C"
