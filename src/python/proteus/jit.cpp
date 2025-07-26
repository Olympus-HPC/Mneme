#include "../llvm/core.h"
#include "mneme/MnemeUtils.hpp"
#include "llvm-c/Core.h"
#include "llvm/IR/Module.h"
#include <chrono>
#include <iostream>
#include <llvm-c/Types.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/Module.h>
#include <mneme/MnemeLogger.hpp>
#include <proteus/CompilerInterfaceTypes.h>
#include <proteus/CoreLLVM.hpp>
#include <proteus/CoreLLVMDevice.hpp>
#include <proteus/Hashing.hpp>

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
                   const char *OptLevel, unsigned CodegenOptLevel) {
  auto *M = llvm::unwrap(Mod);
  auto start = std::chrono::high_resolution_clock::now();
  optimizeIR(*M, DeviceArch, OptLevel, CodegenOptLevel);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> duration = end - start;
  float seconds = duration.count();
  LOG_DEBUG("Middle end compilation took {} \n", seconds);
}

API_EXPORT(LLVMMemoryBufferRef)
ProteusPY_codeGenObject(LLVMModuleRef Mod, const char *DeviceArch, bool use_rtc,
                        unsigned CodegenOptLevel) {
  llvm::SmallPtrSet<void *, 8> GlobalLinkedBinaries;
  auto *M = llvm::unwrap(Mod);
  auto start = std::chrono::high_resolution_clock::now();
  auto DeviceObject = proteus::codegenObject(
      *M, DeviceArch, GlobalLinkedBinaries, use_rtc, CodegenOptLevel);
  auto end = std::chrono::high_resolution_clock::now();

  // Calculate duration and convert to seconds as float
  std::chrono::duration<float> duration = end - start;
  float seconds = duration.count();
  LOG_DEBUG("Backend compilation took {} \n", seconds);
  if (!DeviceObject) {
    std::cout << "Device Object is nullptr\n";
    return nullptr;
  }
  auto *ptr = DeviceObject.release();
  return wrap(ptr);
}

API_EXPORT(LLVMModuleRef)
ProteusPY_linkModules(const char **LLVMIRFiles, int size,
                      LLVMContextRef context, bool prune_flag=true, bool internalize_flag=true) {
  auto Ctx = unwrap(context);
  llvm::SmallVector<std::unique_ptr<llvm::Module>> RecordedModules;
  for (int i = 0; i < size; i++) {
    auto Fn = LLVMIRFiles[i];
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Buffer =
        llvm::MemoryBuffer::getFile(Fn);
    if (!Buffer)
      LOG_FATAL("Error with loading file {}\n Error Code:", Fn,
                Buffer.getError().message());

    llvm::Expected<std::unique_ptr<llvm::Module>> ModuleOrErr =
        llvm::parseBitcodeFile(Buffer->get()->getMemBufferRef(), *Ctx);

    if (!ModuleOrErr)
      LOG_FATAL("Error parsing bitcode: {}",
                llvm::toString(ModuleOrErr.takeError()));

    RecordedModules.emplace_back(std::move(ModuleOrErr.get()));
  }
  
  if(prune_flag) {
    // Call prune here
    // loop through recorded modules here?
  }
  auto Mod = proteus::linkModules(*unwrap(context), RecordedModules);

  if(internalize_flag) {
    // Call internalize here
    // loop through recorded modules here? what's the kernelsym arg?
  }

  return wrap(Mod.release());
}

API_EXPORT(uint64_t)
ProteusPY_specializeArguments(LLVMModuleRef Mod, const uint64_t StaticHash,
                              const char *KernelName, void **KernelArgs,
                              int NumArgs, int *SpecializeIndexes,
                              int NumSpecializations) {
  auto *M = llvm::unwrap(Mod);
  auto *F = M->getFunction(KernelName);
  SmallVector<int32_t> RCTypes(NumSpecializations);
  SmallVector<proteus::RuntimeConstant> RCVec;
  const ArrayRef<int32_t> RCIndices(SpecializeIndexes, NumSpecializations);

  for (int i = 0; i < NumSpecializations; i++) {
    RCTypes[i] = proteus::convertTypeToRuntimeConstantType(
        F->getArg(SpecializeIndexes[i])->getType());
  }
  proteus::getRuntimeConstantValues(KernelArgs, RCIndices, RCTypes, RCVec);

  TransformArgumentSpecialization::transform(*M, *F, RCIndices, RCVec);

  auto Hash = hash(StaticHash, StringRef(KernelName), RCVec);
  return Hash.getValue();
}

API_EXPORT(uint64_t)
ProteusPY_specializeDims(LLVMModuleRef Mod, uint64_t CurrentHash,
                         const char *KernelName, dim3 GridDim, dim3 BlockDim) {
  auto *M = llvm::unwrap(Mod);
  auto *F = M->getFunction(KernelName);
  proteus::setKernelDims(*M, GridDim, BlockDim);
  auto Hash = hash(CurrentHash, GridDim.x, GridDim.y, GridDim.z, BlockDim.x,
                   BlockDim.y, BlockDim.z);
  return Hash.getValue();
}

API_EXPORT(uint64_t)
ProteusPY_setLaunchBounds(LLVMModuleRef Mod, uint64_t CurrentHash,
                          const char *KernelName, int MaxThreadsPerBlock,
                          int MinBlocksPerSM) {
  auto *M = llvm::unwrap(Mod);
  auto *F = M->getFunction(KernelName);
  auto Hash = hash(CurrentHash, MaxThreadsPerBlock, MinBlocksPerSM);
  proteus::setLaunchBoundsForKernel(*F, MaxThreadsPerBlock, MinBlocksPerSM);
  return Hash.getValue();
}
}
