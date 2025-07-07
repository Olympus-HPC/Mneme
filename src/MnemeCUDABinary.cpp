#include <llvm/IR/Function.h>

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeDeviceBinary.hpp"

using namespace mneme;
using namespace llvm;

static std::unique_ptr<Module> ExtractCodeFromDeviceModule(LLVMContext &Ctx,
                                                           CUmodule &CUMod,
                                                           StringRef ModuleId) {
  CUdeviceptr DevPtr;
  size_t Bytes;
  auto EC = DeviceTraits<DeviceVendors::CUDA>::DeviceErrorCheck(
      cuModuleGetGlobal(&DevPtr, &Bytes, CUMod, ModuleId.data()));
  if (EC)
    LOG_FATAL("Device Error extracting code\nEC:{}", EC.value());
  LOG_DEBUG("Found bitcode at addr {} with size {}", (void *)DevPtr, Bytes);
  SmallString<4096> DeviceBitcode;
  DeviceBitcode.reserve(Bytes);
  EC = DeviceTraits<DeviceVendors::CUDA>::DeviceErrorCheck(
      cuMemcpyDtoH(DeviceBitcode.data(), DevPtr, Bytes));
  if (EC)
    LOG_FATAL("Device Error copying device code\nEC:{}", EC.value());
  SMDiagnostic Err;
  return parseIR(
      MemoryBufferRef(StringRef(DeviceBitcode.data(), Bytes), ModuleId), Err,
      Ctx);
}

template <>
void MnemeDeviceLinkedBin::ExtractCodeWithRDC<DeviceVendors::CUDA>(
    LLVMContext &Ctx, SmallVector<std::unique_ptr<Module>> &Modules,
    SmallVector<StringRef> &Blobs) {
  CUmodule CUMod;
  auto EC = DeviceTraits<DeviceVendors::CUDA>::DeviceErrorCheck(
      (cuModuleLoadData(&CUMod, FatBinary->Binary)));
  if (EC)
    LOG_FATAL("Could not load binary fatbin");

  for (int I = 0; I < ModuleIds.size(); I++) {
    auto &ModuleID = ModuleIds[I];
    if (ModuleID.size() == 0) {
      // We need to parse the memory blob
      if (FatBinary->PrelinkedFatBins[I] == nullptr)
        LOG_FATAL("Expected Prelinked binary at index {} to not be nullptr", I);
      Blobs.emplace_back(StringRef(
          reinterpret_cast<const char *>(FatBinary->PrelinkedFatBins[I]),
          FatBinary->PrelinkedFatBins[I]->HeaderSize +
              FatBinary->PrelinkedFatBins[I]->FatSize));
    } else {
      Modules.emplace_back(ExtractCodeFromDeviceModule(Ctx, CUMod, ModuleID));
    }
  }
}

template <>
void MnemeDeviceLinkedBin::ExtractCodeWithoutRDC<DeviceVendors::CUDA>(
    LLVMContext &Ctx, SmallVector<std::unique_ptr<Module>> &Modules) {
  if (ModuleIds.size() != 1)
    LOG_FATAL("Expected Non RDC module to contain a single LLVM IR");

  CUmodule CUMod;
  auto EC = DeviceTraits<DeviceVendors::CUDA>::DeviceErrorCheck(
      (cuModuleLoadData(&CUMod, FatBinary->Binary)));
  if (EC)
    LOG_FATAL("Could not load binary fatbin");
  Modules.emplace_back(ExtractCodeFromDeviceModule(Ctx, CUMod, ModuleIds[0]));
  EC = DeviceTraits<DeviceVendors::CUDA>::DeviceErrorCheck(
      cuModuleUnload(CUMod));
  if (EC)
    LOG_FATAL("Could not load binary fatbin");
}

template <>
void MnemeDeviceLinkedBin::FindKernels<DeviceVendors::CUDA>(
    llvm::DenseMap<llvm::StringRef, llvm::Function *> &KernelNameToFunction,
    llvm::Module &M) {
  NamedMDNode *MD = M.getOrInsertNamedMetadata("nvvm.annotations");

  if (!MD)
    return;

  for (auto *Op : MD->operands()) {
    if (Op->getNumOperands() < 2)
      continue;
    MDString *KindID = dyn_cast<MDString>(Op->getOperand(1));
    if (!KindID || KindID->getString() != "kernel")
      continue;

    Function *KernelFn =
        mdconst::dyn_extract_or_null<Function>(Op->getOperand(0));
    if (!KernelFn)
      continue;

    KernelNameToFunction.try_emplace(KernelFn->getName(), KernelFn);
  }
}
