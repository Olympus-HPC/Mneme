#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "core.h"
#include <cstdio>
#include <iostream>
#include <string>

extern "C" {

API_EXPORT(LLVMModuleRef)
LLVMPY_ParseAssembly(LLVMContextRef context, const char *ir,
                     const char **outmsg) {
  using namespace llvm;

  SMDiagnostic error;

  Module *m = parseAssemblyString(ir, error, *unwrap(context)).release();
  if (!m) {
    // Error occurred
    std::string osbuf;
    raw_string_ostream os(osbuf);
    error.print("", os);
    os.flush();
    *outmsg = LLVMPY_CreateString(os.str().c_str());
    return NULL;
  }

  std::cout << "Assembly pointer is " << (void *)m << "\n";
  return wrap(m);
}

} // end extern "C"
