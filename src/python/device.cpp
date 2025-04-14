#include "llvm/core.h"
#include <cstring>
#include <hip/hip_runtime_api.h>
#include <llvm/Support/CBindingWrapping.h>
#include <llvm/Support/MemoryBuffer.h>
#include <mneme/DeviceTraits.hpp>

using namespace mneme;
using namespace llvm;

#ifdef MNEME_ENABLE_HIP
using DeviceVendorTraits = DeviceTraits<DeviceVendors::HIP>;
constexpr DeviceVendors Vendor = DeviceVendors::HIP;
#endif

extern "C" {
API_EXPORT(void *) MnemePY_getDeviceObject(LLVMMemoryBufferRef Buffer) {
  std::cout << "Creating module\n";
  llvm::MemoryBuffer *DeviceObject = llvm::unwrap(Buffer);
  auto VendorModule = DeviceVendorTraits::getDeviceModuleFromImage(
      DeviceObject->getBufferStart());
  // NOTE: HIP modules are effectively a  pointer to some structure.
  // so we can do this. We need to check also what happens in cuda.
  return reinterpret_cast<void *>(VendorModule);
}

API_EXPORT(void) MnemePY_DisposeDeviceObject(void *WModule) {
  std::cout << "Destroying module\n";
  auto VendorModule =
      reinterpret_cast<DeviceVendorTraits::DeviceModule_t>(WModule);
  DeviceVendorTraits::DeviceModuleUnload(VendorModule);
  return;
}

API_EXPORT(void *)
MnemePY_getKernelFunctionFromImage(void *WrappedModule,
                                   const char *KernelName) {
  auto Module =
      reinterpret_cast<DeviceVendorTraits::DeviceModule_t>(WrappedModule);
  std::string KName(KernelName);
  auto Func = DeviceVendorTraits::getKernelFunctionFromImage(Module, KName);
  // NOTE: HIP modules are effectively a  pointer to some structure.
  // so we can do this. We need to check also what happens in cuda.
  auto DevFunc = reinterpret_cast<void *>(Func);
  return DevFunc;
}

API_EXPORT(void)
MnemePY_launchKernelFunction(void *Func, dim3 Grid, dim3 Block) {
  std::cout << "Received function " << Func << "\n";
  auto DevFunc = reinterpret_cast<DeviceVendorTraits::DeviceFunction_t>(Func);
  int Arg = 42;
  void *KernelArgs[] = {&Arg};

  auto EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::launchKernelFunction(DevFunc, Grid, Block, KernelArgs,
                                               0, 0));
  if (EC)
    FATAL_ERROR("Error When Launching Kernel: " + EC.value());

  EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::DeviceSynchronize());

  if (EC)
    FATAL_ERROR("Error When Launching Kernel: " + EC.value());
}

const char *MnemePy_getDeviceArch() {
  auto Arch = DeviceVendorTraits::GetDeviceArch();
  auto *ret = strdup(Arch.c_str());
  std::cout << "Got pointer " << (void *)ret << "\n";
  return ret;
}
}
