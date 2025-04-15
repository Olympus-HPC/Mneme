#include "../llvm/core.h"
#include "mneme/Utils.hpp"
#include "llvm-c/Core.h"
#include "llvm/IR/Module.h"
#include <iostream>
#include <llvm-c/Types.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/Module.h>
#include <proteus/CoreLLVM.hpp>
#include <proteus/CoreLLVMDevice.hpp>

using namespace proteus;

extern "C" {
API_EXPORT(void) ProteusPY_pruneIR(LLVMModuleRef Mod) {
  pruneIR(*llvm::unwrap(Mod));
}

API_EXPORT(void)
ProteusPY_internalize(LLVMModuleRef Mod, const char *KernelSym) {
  auto *M = llvm::unwrap(Mod);
  internalize(*M, KernelSym);
}

API_EXPORT(void)
ProteusPY_optimize(LLVMModuleRef Mod, const char *DeviceArch,
                   const char OptLevel, unsigned CodegenOptLevel) {
  auto *M = llvm::unwrap(Mod);
  optimizeIR(*M, DeviceArch, OptLevel, CodegenOptLevel);
}

API_EXPORT(LLVMMemoryBufferRef)
ProteusPY_codeGenObject(LLVMModuleRef Mod, const char *DeviceArch) {
  llvm::SmallPtrSet<void *, 8> GlobalLinkedBinaries;
  auto *M = llvm::unwrap(Mod);
  auto DeviceObject =
      proteus::codegenObject(*M, DeviceArch, GlobalLinkedBinaries);
  if (!DeviceObject)
    return nullptr;
  auto *ptr = DeviceObject.release();
  return wrap(ptr);
}

API_EXPORT(LLVMModuleRef)
ProteusPY_linkModules(const char **LLVMIRFiles, int size,
                      LLVMContextRef context) {
  auto Ctx = unwrap(context);
  llvm::SmallVector<std::unique_ptr<llvm::Module>> RecordedModules;
  for (int i = 0; i < size; i++) {
    auto Fn = LLVMIRFiles[i];
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Buffer =
        llvm::MemoryBuffer::getFile(Fn);
    if (!Buffer)
      FATAL_ERROR("Error with loading file " + Fn +
                  "\n Error Code:" + Buffer.getError().message());

    llvm::Expected<std::unique_ptr<llvm::Module>> ModuleOrErr =
        llvm::parseBitcodeFile(Buffer->get()->getMemBufferRef(), *Ctx);

    if (!ModuleOrErr)
      FATAL_ERROR("Error parsing bitcode: " +
                  llvm::toString(ModuleOrErr.takeError()));

    RecordedModules.emplace_back(std::move(ModuleOrErr.get()));
  }

  auto Mod = proteus::linkModules(*unwrap(context), RecordedModules);
  return wrap(Mod.release());
}
}
