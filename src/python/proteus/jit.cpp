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
#include <proteus/CompilerInterfaceRuntimeConstantInfo.h>
#include <proteus/CompilerInterfaceTypes.h>
#include <proteus/CoreLLVM.hpp>
#include <proteus/CoreLLVMDevice.hpp>
#include <proteus/Hashing.hpp>

using namespace proteus;

namespace {
inline std::optional<CodegenOption> fromString(std::string str) {
  if (str.compare("rtc") == 0) {
    return CodegenOption::RTC;
  } else if (str.compare("serial") == 0) {
    return CodegenOption::Serial;
  } else if (str.compare("parallel") == 0) {
    return CodegenOption::Parallel;
  }
  return std::nullopt;
}

} // namespace

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
  LOG_DEBUG("Middle end compilation took {}", seconds);
}

API_EXPORT(LLVMMemoryBufferRef)
ProteusPY_codeGenObject(LLVMModuleRef Mod, const char *DeviceArch,
                        const char *rtc, unsigned CodegenOptLevel) {
  auto proteus_rtc = fromString(rtc);
  if (!proteus_rtc)
    LOG_FATAL("Unknown RTC value of {}", rtc);

  llvm::SmallPtrSet<void *, 8> GlobalLinkedBinaries;
  auto *M = llvm::unwrap(Mod);
  auto start = std::chrono::high_resolution_clock::now();
  auto DeviceObject = proteus::codegenObject(
      *M, DeviceArch, GlobalLinkedBinaries, proteus_rtc.value());
  auto end = std::chrono::high_resolution_clock::now();

  // Calculate duration and convert to seconds as float
  std::chrono::duration<float> duration = end - start;
  float seconds = duration.count();
  LOG_DEBUG("Backend compilation took {}", seconds);
  if (!DeviceObject) {
    LOG_WARN("Device Object is nullptr");
    return nullptr;
  }
  auto *ptr = DeviceObject.release();
  return wrap(ptr);
}

API_EXPORT(LLVMModuleRef)
ProteusPY_linkModules(const char **LLVMIRFiles, int size,
                      LLVMContextRef context, const char *KernelSym,
                      bool prune_flag = true, bool internalize_flag = true) {
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

    if (prune_flag) {
      pruneIR(*ModuleOrErr->get());
    }

    RecordedModules.emplace_back(std::move(ModuleOrErr.get()));
  }

  auto Mod = proteus::linkModules(*unwrap(context), std::move(RecordedModules));

  if (internalize_flag) {
    internalize(*Mod.get(), KernelSym);
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
  SmallVector<std::unique_ptr<RuntimeConstantInfo>, 64> RCStorage;

  for (int i = 0; i < NumSpecializations; i++) {
    RCStorage.emplace_back(std::make_unique<RuntimeConstantInfo>(
        proteus::convertTypeToRuntimeConstantType(
            F->getArg(SpecializeIndexes[i])->getType()),
        SpecializeIndexes[i]));
  }

  SmallVector<RuntimeConstant> RCVec;
  RCVec.reserve(RCStorage.size());

  for (const auto &RCInfo : RCStorage) {
    PROTEUS_DBG(Logger::logs("proteus")
                << "RC Index " << RCInfo->ArgInfo.Pos << " Type "
                << toString(RCInfo->ArgInfo.Type) << " ");

    RCVec.emplace_back(dispatchGetRuntimeConstantValue(KernelArgs, *RCInfo));
  }

  TransformArgumentSpecialization::transform(*M, *F, RCVec);

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
ProteusPY_specializeDimsAssume(LLVMModuleRef Mod, uint64_t CurrentHash,
                         const char *KernelName, dim3 GridDim, dim3 BlockDim) {
  auto *M = llvm::unwrap(Mod);
  auto *F = M->getFunction(KernelName);
  proteus::setKernelDimsAssume(*M, GridDim, BlockDim);
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
  LOG_INFO("Was called with {} and {}", MaxThreadsPerBlock, MinBlocksPerSM);
  proteus::setLaunchBoundsForKernel(*F, MaxThreadsPerBlock, MinBlocksPerSM);
  return Hash.getValue();
}
}
