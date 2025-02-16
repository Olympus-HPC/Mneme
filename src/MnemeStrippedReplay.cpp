// RUN: ./daxpy.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./daxpy.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
#include "proteus/Error.h"
#include <cstddef>
#include <cstdlib>
#include <iostream>

#include <proteus/CoreLLVM.hpp>
#include <proteus/CoreLLVMDevice.hpp>

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Support/InitLLVM.h>
#include <string>
#ifdef PROTEUS_ENABLE_HIP
#include "proteus/CoreLLVMHIP.hpp"
#elif defined(PROTEUS_ENABLE_CUDA)
#include "proteus/CoreLLVMCUDA.hpp"
#endif

// TODO: To run execute <exe> _Z3fooi

int main(int argc, char **argv) {
  proteus::InitNativeTarget();
  std::string DeviceArch;
#ifdef PROTEUS_ENABLE_HIP
  hipDeviceProp_t DevProp;
  proteus::InitAMDGPUTarget();
  proteusHipErrCheck(hipGetDeviceProperties(&DevProp, 0));

  DeviceArch = DevProp.gcnArchName;
  DeviceArch = DeviceArch.substr(0, DeviceArch.find_first_of(":"));

#elif defined(PROTEUS_ENABLE_CUDA)
  proteus::InitNVPTXTarget();
  CUdevice CUDev;
  CUcontext CUCtx;

  proteusCuErrCheck(cuInit(0));

  CUresult CURes = cuCtxGetDevice(&CUDev);
  if (CURes == CUDA_ERROR_INVALID_CONTEXT or !CUDev)
    // TODO: is selecting device 0 correct?
    proteusCuErrCheck(cuDeviceGet(&CUDev, 0));

  proteusCuErrCheck(cuCtxGetCurrent(&CUCtx));
  if (!CUCtx)
    proteusCuErrCheck(cuCtxCreate(&CUCtx, 0, CUDev));

  int CCMajor;
  proteusCuErrCheck(cuDeviceGetAttribute(
      &CCMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, CUDev));
  int CCMinor;
  proteusCuErrCheck(cuDeviceGetAttribute(
      &CCMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, CUDev));
  DeviceArch = "sm_" + std::to_string(CCMajor * 10 + CCMinor);

#endif
  std::string bitcodeFN(argv[1]);
  llvm::LLVMContext Ctx;
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Buffer =
      llvm::MemoryBuffer::getFile(bitcodeFN);
  if (!Buffer)
    PROTEUS_FATAL_ERROR("Error with loading file " + bitcodeFN +
                        "\n Error Code:" + Buffer.getError().message());

  llvm::Expected<std::unique_ptr<llvm::Module>> ModuleOrErr =
      llvm::parseBitcodeFile(Buffer->get()->getMemBufferRef(), Ctx);

  if (!ModuleOrErr)
    PROTEUS_FATAL_ERROR("Error parsing bitcode: " +
                        llvm::toString(ModuleOrErr.takeError()));

  llvm::SmallPtrSet<void *, 8> GlobalLinkedBinaries;
  auto Mod = std::move(ModuleOrErr.get());
  proteus::pruneIR(*Mod);
  auto KernelFunc = Mod->getFunction(argv[2]);
  proteus::internalize(*Mod, KernelFunc->getName());
  proteus::optimizeIR(*Mod, DeviceArch, '1', 1);
  auto DeviceObject =
      proteus::codegenObject(*Mod, DeviceArch, GlobalLinkedBinaries);
  std::cout << "Success \n";
}
