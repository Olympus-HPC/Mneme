// Copyright (c) 2021-2024. LLNL-CODE-2000766 and Mneme Contributors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifdef ICMP_NE
#undef ICMP_NE
#endif
#include <cstdint>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/CommandLine.h>

#include <proteus/CoreLLVM.h>
#include <proteus/CoreLLVMDevice.h>

#include <llvm/Support/InitLLVM.h>
#ifdef MNEME_ENABLE_HIP
#include <proteus/CoreLLVMHIP.h>
#elif defined(MNEME_ENABLE_CUDA)
#else
#error "Please define MNEME_ENABLE_HIP or MNEME_ENABLE_CUDA"
#endif

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeJITProteus.hpp"
#include "mneme/MnemeLLVMUtils.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeReplay.hpp"

#include <llvm/IR/Module.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>

using namespace mneme;
using namespace llvm;

#ifdef MNEME_ENABLE_HIP
using DeviceVendorTraits = DeviceTraits<DeviceVendors::HIP>;
constexpr DeviceVendors Vendor = DeviceVendors::HIP;
#elif defined(MNEME_ENABLE_CUDA)
using DeviceVendorTraits = DeviceTraits<DeviceVendors::CUDA>;
constexpr DeviceVendors Vendor = DeviceVendors::CUDA;
#endif

static cl::OptionCategory MnemeCategory("Mneme Tool Options",
                                        "Mneme CLI options.");

static cl::opt<std::string> MnemeJson(
    "mneme-replay-json",
    cl::desc("The json file containing metadata for the recorded kernels"),
    cl::Required, cl::cat(MnemeCategory));

static cl::opt<std::string>
    MnemeKernelHash("mneme-replay-hash",
                    cl::desc("The Kernel Hash of the Recorded kernels"),
                    cl::Required, cl::cat(MnemeCategory));

static cl::opt<char> MnemeMiddleOptLevel(
    "middle-opt-level",
    cl::desc("The optimization level to use when optimizing IR"), cl::init('1'),
    cl::cat(MnemeCategory));

static cl::opt<unsigned> MnemeBackendOptLevel(
    "backend-opt-level",
    cl::desc("The optimization level to use when optimizing IR"), cl::init(1),
    cl::cat(MnemeCategory));

static cl::alias MnemeShortMiddleOptLevel(
    "MO", cl::desc("The optimization level to use when optimizing IR"),
    cl::aliasopt(MnemeMiddleOptLevel), cl::cat(MnemeCategory));

static cl::alias MnemeShortBackEndOptLevel(
    "BO", cl::desc("The optimization level to use when optimizing IR"),
    cl::aliasopt(MnemeBackendOptLevel), cl::cat(MnemeCategory));

static cl::opt<int>
    MnemeRepeats("repeats",
                 cl::desc("The number of repeat executions for every kernel"),
                 cl::init(3), cl::cat(MnemeCategory));

void writeIRToFile(const llvm::Module &M, const std::string &Filename) {
  std::error_code EC;
  llvm::raw_fd_ostream File(Filename, EC, llvm::sys::fs::OF_Text);

  if (EC) {
    llvm::errs() << "Error opening file: " << EC.message() << "\n";
    return;
  }

  M.print(File, nullptr); // Print the module's IR to the file
}

int main(int argc, char *argv[]) {
  cl::HideUnrelatedOptions(MnemeCategory);
  cl::ParseCommandLineOptions(argc, argv, "GPU Replay Tool\n");
  proteus::InitLLVMTargets Init;
  // NOTE: There is a weird interaction of proteus with LLVM and the CLI option
  // manager is initialized (at least) twice. This has as a side effect to reset
  // variables to their default vvalues.
  int _MnemeRepeats = MnemeRepeats;

  LOG_INFO("using mneme db-file {} and dynamic instance {}", MnemeJson,
           MnemeKernelHash);
  auto Arch = DeviceVendorTraits::GetDeviceArch();
  LOG_INFO("Detected system device Architecture is: {}", Arch);

  ReplayInstance<Vendor> RInstance(MnemeJson, MnemeKernelHash);
  llvm::LLVMContext Ctx;
  auto Modules = RInstance.loadModules(Ctx);
  auto Mod = proteus::linkModules(Ctx, std::move(Modules));
  proteus::pruneIR(*Mod, false);
  pruneMnemeGlobals(*Mod);
  auto ReplayKernelFunc = Mod->getFunction(RInstance.getKernelName());
  // TODO: Internalize is too aggresive as is now. We will need to either write
  // our own, or wait for proteus to expose a more generic interface.
  // proteus::internalize(*Mod, ReplayKernelFunc->getName());

  // TODO: Here I need to write the module. With a name FnName.StaticHash.bc
  // Before optimization, to make sure we can train models in a "generic" way.

  LOG_INFO("Optimizing Kernel with Middle-OptLevel {} and BackEnd-OptLevel {}",
           MnemeMiddleOptLevel.getValue(), MnemeBackendOptLevel.getValue());
  proteus::optimizeIR(*Mod, Arch, MnemeMiddleOptLevel.getValue(),
                      MnemeBackendOptLevel.getValue());

  auto RecordedGrid = RInstance.getRecordedGrid();
  auto RecordedBlock = RInstance.getRecordedBlock();
  proteus::setLaunchBoundsForKernel(
      *ReplayKernelFunc, RecordedBlock.x * RecordedBlock.y * RecordedBlock.z);
  proteus::runCleanupPassPipeline(*Mod);
  SmallPtrSet<void *, 8> GlobalLinkedBinaries;

  auto DeviceObject = proteus::codegenObject(*Mod, Arch, GlobalLinkedBinaries);
  auto VendorModule = DeviceVendorTraits::getDeviceModuleFromImage(
      DeviceObject->getBufferStart());

  RInstance.initializeDeviceMemory();
  LOG_DEBUG("Initialized Device Memory");
  RInstance.initializeGlobals(VendorModule);
  LOG_DEBUG("Initialized Device Globals");

  auto Func = DeviceVendorTraits::getKernelFunctionFromImage(
      VendorModule, RInstance.getKernelName());

  bool verify = true;
  DeviceVendorTraits::DeviceStream_t ReplayStream;
  DeviceVendorTraits::DeviceEvent_t StartEvent, EndEvent;
  auto EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::DeviceStreamCreate(&ReplayStream));
  if (EC)
    LOG_FATAL("Error when creating a stream for replay\n" + EC.value());

  EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::deviceEventCreate(&StartEvent));

  if (EC)
    LOG_FATAL("Error when creating start event for replay\n" + EC.value());

  EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::deviceEventCreate(&EndEvent));

  if (EC)
    LOG_FATAL("Error when creating end event for replay\n" + EC.value());

  std::vector<float> timings;
  for (int i = 0; i < _MnemeRepeats; i++) {
    float elapsedTime;
    LOG_DEBUG("Run {}/{}", i + 1, _MnemeRepeats);

    auto Args = RInstance.getArgs();

    DeviceVendorTraits::DeviceErrorCheck(
        DeviceVendorTraits::deviceEventRecord(StartEvent, ReplayStream));
    if (EC)
      LOG_FATAL("Error when recording event " + EC.value());

    EC = DeviceVendorTraits::DeviceErrorCheck(
        DeviceVendorTraits::launchKernelFunction(
            Func, RecordedGrid, RecordedBlock, Args,
            RInstance.getSharedMemSize(), ReplayStream));
    if (EC)
      LOG_FATAL("Error When Launching Kernel: " + EC.value());

    DeviceVendorTraits::DeviceErrorCheck(
        DeviceVendorTraits::deviceEventRecord(EndEvent, ReplayStream));
    if (EC)
      LOG_FATAL("Error when recording event " + EC.value());

    EC = DeviceVendorTraits::DeviceErrorCheck(
        DeviceVendorTraits::deviceEventSynchronize(EndEvent));
    if (EC)
      LOG_FATAL("Error when synchronizing event " + EC.value());

    EC = DeviceVendorTraits::DeviceErrorCheck(
        DeviceVendorTraits::deviceEventElapsedTime(&elapsedTime, StartEvent,
                                                   EndEvent));
    if (EC)
      LOG_FATAL("Error when recording event " + EC.value());

    std::cout << "Kernel execution time: " << elapsedTime << " ms\n";
    timings.push_back(elapsedTime);

    EC = DeviceVendorTraits::DeviceErrorCheck(
        DeviceVendorTraits::DeviceStreamSynchronize(ReplayStream));

    if (EC)
      LOG_FATAL("Error When synchronizing with kernel stream: " + EC.value());

    verify &= RInstance.isMemorySame();
    RInstance.reset();
  }

  float sum = std::accumulate(timings.begin(), timings.end(), 0.0);
  float mean = sum / timings.size();
  // Compute the standard deviation
  float sq_sum =
      std::inner_product(timings.begin(), timings.end(), timings.begin(), 0.0);
  float std_dev = std::sqrt(sq_sum / timings.size() - mean * mean);
  std::cout << "Mean execution time (ms): " << mean
            << " standard Deviation: " << std_dev << "\n";

  EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::deviceStreamDestroy(ReplayStream));
  if (EC)
    LOG_FATAL("Error when destroying stream for replay\n" + EC.value());

  EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::deviceEventDestroy(StartEvent));

  if (EC)
    LOG_FATAL("Error when destroying start event for replay\n" + EC.value());

  EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::deviceEventDestroy(EndEvent));

  if (EC)
    LOG_FATAL("Error when destroying end event for replay\n" + EC.value());

  if (verify)
    std::cout << "Results Match" << "\n";
  else
    std::cout << "Results DO NOT Match" << "\n";

  RInstance.releaseMemory();
  return !verify;
}
