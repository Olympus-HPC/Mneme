#include "llvm/core.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
  // NOTE: Device modules are effectively a  pointer to some structure.
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
  // NOTE: Device modules are effectively a  pointer to some structure.
  // so we can do this. We need to check also what happens in cuda.
  auto DevFunc = reinterpret_cast<void *>(Func);
  LOG_DEBUG("Device Function pointer is {}", DevFunc);
  return DevFunc;
}

API_EXPORT(int)
MnemePy_getRegisterUsage(void *Func) {
  int UsedRegisters = -1;
  auto DevFunc = reinterpret_cast<DeviceVendorTraits::DeviceFunction_t>(Func);
  auto EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::deviceGetAttribute(
          DevFunc, FuncAttributes::REGISTER_USAGE, UsedRegisters));
  if (EC)
    LOG_FATAL("Error when reading register usage: " + EC.value());
  return UsedRegisters;
}

API_EXPORT(int)
MnemePy_getLocalMemUsage(void *Func) {
  int LocalMemUsage = -1;
  auto DevFunc = reinterpret_cast<DeviceVendorTraits::DeviceFunction_t>(Func);
  auto EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::deviceGetAttribute(
          DevFunc, FuncAttributes::LOCALMEM_USAGE, LocalMemUsage));
  if (EC)
    LOG_FATAL("Error when reading register usage: " + EC.value());
  return LocalMemUsage;
}

API_EXPORT(int)
MnemePy_getConstMemUsage(void *Func) {
  int ConstMemUsage = 0;
  auto DevFunc = reinterpret_cast<DeviceVendorTraits::DeviceFunction_t>(Func);
  auto EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::deviceGetAttribute(
          DevFunc, FuncAttributes::CONSTMEM_USAGE, ConstMemUsage));
  if (EC)
    LOG_FATAL("Error when reading register usage: " + EC.value());
  return ConstMemUsage;
}

API_EXPORT(void)
MnemePy_launchKernelFunction(void *Func, dim3 Grid, dim3 Block) {
  auto DevFunc = reinterpret_cast<DeviceVendorTraits::DeviceFunction_t>(Func);
  int Arg = 42;
  void *KernelArgs[] = {&Arg};

  auto EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::launchKernelFunction(DevFunc, Grid, Block, KernelArgs,
                                               0, 0));
  if (EC)
    LOG_FATAL("Error When Launching Kernel: " + EC.value());

  EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::DeviceSynchronize());

  if (EC)
    LOG_FATAL("Error When Launching Kernel: " + EC.value());
}

API_EXPORT(const char *) MnemePy_getDeviceArch() {
  auto Arch = DeviceVendorTraits::GetDeviceArch();
  auto *ret = strdup(Arch.c_str());
  return ret;
}

API_EXPORT(void)
MnemePy_profile(void *WrappedModule, void *Func, dim3 Grid, dim3 Block,
                MnemeDeviceMemStateRef Prologue,
                MnemeDeviceMemStateRef Epilogue, int SharedMemSize, int repeats,
                const char *ResetModeName) {
  auto VendorModule =
      reinterpret_cast<DeviceVendorTraits::DeviceModule_t>(WrappedModule);

  int DeviceId;
  auto EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::getDevice(DeviceId));
  if (EC)
    LOG_FATAL("Error when requesting active device\n", EC.value());

  LOG_INFO("Executing replay run on device {}", DeviceId);

  DeviceVendorTraits::DeviceStream_t ReplayStream;
  auto DevFunc = reinterpret_cast<DeviceVendorTraits::DeviceFunction_t>(Func);
  auto *ProState = unwrap(Prologue)->asPrologue();
  auto *EpiState = unwrap(Epilogue)->asEpilogue();
  if (!ProState || !EpiState)
    LOG_FATAL("MnemePy_profile expects a prologue and an epilogue state");
  std::string ResetModeString =
      ResetModeName == nullptr ? "" : std::string(ResetModeName);
  ReplayResetMode ResetMode;
  if (ResetModeString.empty()) {
    const char *EnvMode = std::getenv("MNEME_REPLAY_RESET_MODE");
    if (EnvMode != nullptr) {
      ResetMode = parseReplayResetMode(EnvMode);
    } else if (MnemeSnapshot<Vendor>::isDiffSnapshotFile(
                   EpiState->getSnapshotPath())) {
      ResetMode = ReplayResetMode::Diff;
    } else {
      ResetMode = ReplayResetMode::Bytes;
    }
  } else {
    ResetMode = parseReplayResetMode(ResetModeString);
  }
  bool TraceResetTiming = std::getenv("MNEME_REPLAY_RESET_TIMING") != nullptr;

  EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::DeviceStreamCreate(&ReplayStream));
  if (EC)
    LOG_FATAL("Error when creating a stream for replay\n" + EC.value());

  ProState->initializeGlobals(VendorModule);
  ProState->prepareResetPlan(*EpiState, ResetMode);

  for (int i = 0; i < repeats; i++) {
    LOG_DEBUG("Run {}/{}", i + 1, repeats);

    auto IterStart = std::chrono::steady_clock::now();
    auto ResetStart = std::chrono::steady_clock::now();
    if (i == 0)
      ProState->reset();
    else
      ProState->reset(ResetMode, ReplayStream);
    auto ResetStop = std::chrono::steady_clock::now();
    if (TraceResetTiming) {
      auto ResetNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         ResetStop - ResetStart)
                         .count();
      bool UsedDiff = i != 0 && ResetMode == ReplayResetMode::Diff;
      const char *ResetKind = "full";
      if (UsedDiff)
        ResetKind = "diff";
      std::fprintf(stderr,
                   "MNEME_RESET_TIMING repeat=%d/%d kind=%s ns=%lld\n",
                   i + 1, repeats, ResetKind, static_cast<long long>(ResetNs));
    }
    auto Args = ProState->getArgs();

    auto PreSyncStart = std::chrono::steady_clock::now();
    EC = DeviceVendorTraits::DeviceErrorCheck(
        DeviceVendorTraits::DeviceStreamSynchronize(ReplayStream));
    auto PreSyncStop = std::chrono::steady_clock::now();

    if (EC)
      LOG_FATAL("Error when synchronizing device stream " + EC.value());

    auto LaunchStart = std::chrono::steady_clock::now();
    EC = DeviceVendorTraits::DeviceErrorCheck(
        DeviceVendorTraits::launchKernelFunction(DevFunc, Grid, Block, Args,
                                                 SharedMemSize, ReplayStream));
    auto LaunchStop = std::chrono::steady_clock::now();
    if (EC)
      LOG_FATAL("Error When Launching Kernel: " + EC.value());

    auto PostSyncStart = std::chrono::steady_clock::now();
    EC = DeviceVendorTraits::DeviceErrorCheck(
        DeviceVendorTraits::DeviceStreamSynchronize(ReplayStream));
    auto PostSyncStop = std::chrono::steady_clock::now();

    if (EC)
      LOG_FATAL("Error When synchronizing with kernel stream: " + EC.value());

    if (TraceResetTiming) {
      auto IterNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        PostSyncStop - IterStart)
                        .count();
      auto PreSyncNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           PreSyncStop - PreSyncStart)
                           .count();
      auto LaunchNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          LaunchStop - LaunchStart)
                          .count();
      auto PostSyncNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            PostSyncStop - PostSyncStart)
                            .count();
      bool UsedDiff = i != 0 && ResetMode == ReplayResetMode::Diff;
      const char *ReplayKind = "full";
      if (UsedDiff)
        ReplayKind = "diff";
      std::fprintf(stderr,
                   "MNEME_REPLAY_TIMING repeat=%d/%d kind=%s iter_ns=%lld "
                   "pre_sync_ns=%lld launch_ns=%lld post_sync_ns=%lld\n",
                   i + 1, repeats, ReplayKind, static_cast<long long>(IterNs),
                   static_cast<long long>(PreSyncNs),
                   static_cast<long long>(LaunchNs),
                   static_cast<long long>(PostSyncNs));
    }
  }

  EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::DeviceSynchronize());

  if (EC)
    LOG_FATAL("Error when synchronizing device" + EC.value());
}

API_EXPORT(int) MnemePy_getNumArgs(MnemeDeviceMemStateRef WState) {
  auto State = unwrap(WState);
  return State->getNumArgs();
}

API_EXPORT(void **) MnemePy_getArgs(MnemeDeviceMemStateRef WState) {
  auto State = unwrap(WState);
  return reinterpret_cast<void **>(State->getArgs());
}

API_EXPORT(int) MnemePy_getDeviceCount() {
  int DevCount;
  auto EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::getDeviceCount(DevCount));
  if (EC)
    LOG_FATAL("Error when getting the number of available devices\nEC: {}",
              EC.value());
  return DevCount;
}

API_EXPORT(void) MnemePy_setDevice(int DevId) {
  auto EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::setDevice(DevId));
  if (EC)
    LOG_FATAL("Error when setting the device id\nEC: {}", EC.value());
}
}
