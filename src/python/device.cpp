#include "llvm/core.h"
#include <cstring>
#include <hip/hip_runtime_api.h>
#include <llvm/Support/CBindingWrapping.h>
#include <llvm/Support/MemoryBuffer.h>
#include <mneme/MnemePython.hpp>

using namespace mneme;
using namespace mneme::python;
using namespace llvm;

extern "C" {
API_EXPORT(void *) MnemePY_getDeviceObject(LLVMMemoryBufferRef Buffer) {
  llvm::MemoryBuffer *DeviceObject = llvm::unwrap(Buffer);
  auto VendorModule = DeviceVendorTraits::getDeviceModuleFromImage(
      DeviceObject->getBufferStart());
  // NOTE: HIP modules are effectively a  pointer to some structure.
  // so we can do this. We need to check also what happens in cuda.
  return reinterpret_cast<void *>(VendorModule);
}

API_EXPORT(void) MnemePY_DisposeDeviceObject(void *WModule) {
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

API_EXPORT(const char *) MnemePy_getDeviceArch() {
  auto Arch = DeviceVendorTraits::GetDeviceArch();
  auto *ret = strdup(Arch.c_str());
  return ret;
}

API_EXPORT(void)
MnemePy_profile(void *WrappedModule, void *Func, dim3 Grid, dim3 Block,
                MnemeDeviceMemStateRef MemState, int SharedMemSize, int repeats,
                float *time) {
  auto VendorModule =
      reinterpret_cast<DeviceVendorTraits::DeviceModule_t>(WrappedModule);

  DeviceVendorTraits::DeviceStream_t ReplayStream;
  DeviceVendorTraits::DeviceEvent_t StartEvent, EndEvent;
  auto DevFunc = reinterpret_cast<DeviceVendorTraits::DeviceFunction_t>(Func);
  auto State = unwrap(MemState);
  auto EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::DeviceStreamCreate(&ReplayStream));
  if (EC)
    FATAL_ERROR("Error when creating a stream for replay\n" + EC.value());

  EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::deviceEventCreate(&StartEvent));

  if (EC)
    FATAL_ERROR("Error when creating start event for replay\n" + EC.value());

  EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::deviceEventCreate(&EndEvent));

  if (EC)
    FATAL_ERROR("Error when creating end event for replay\n" + EC.value());

  State->initializeGlobals(VendorModule);

  for (int i = 0; i < repeats; i++) {
    float elapsedTime;
    LOG_DEBUG("Run {}/{}", i + 1, repeats);

    auto Args = State->getArgs();
    State->reset();

    DeviceVendorTraits::DeviceErrorCheck(
        DeviceVendorTraits::deviceEventRecord(StartEvent, ReplayStream));
    if (EC)
      FATAL_ERROR("Error when recording event " + EC.value());

    EC = DeviceVendorTraits::DeviceErrorCheck(
        DeviceVendorTraits::launchKernelFunction(
            DevFunc, Grid, Block, Args.get(), SharedMemSize, ReplayStream));
    if (EC)
      FATAL_ERROR("Error When Launching Kernel: " + EC.value());

    DeviceVendorTraits::DeviceErrorCheck(
        DeviceVendorTraits::deviceEventRecord(EndEvent, ReplayStream));
    if (EC)
      FATAL_ERROR("Error when recording event " + EC.value());

    EC = DeviceVendorTraits::DeviceErrorCheck(
        DeviceVendorTraits::deviceEventSynchronize(EndEvent));
    if (EC)
      FATAL_ERROR("Error when synchronizing event " + EC.value());

    EC = DeviceVendorTraits::DeviceErrorCheck(
        DeviceVendorTraits::deviceEventElapsedTime(&elapsedTime, StartEvent,
                                                   EndEvent));
    if (EC)
      FATAL_ERROR("Error when recording event " + EC.value());

    time[i] = elapsedTime;

    EC = DeviceVendorTraits::DeviceErrorCheck(
        DeviceVendorTraits::DeviceStreamSynchronize(ReplayStream));

    if (EC)
      FATAL_ERROR("Error When synchronizing with kernel stream: " + EC.value());
  }
}
}
