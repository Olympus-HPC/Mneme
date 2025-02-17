#include <hip/hip_runtime.h>

#ifdef ICMP_NE
#undef ICMP_NE
#endif

#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/CommandLine.h>

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeJITProteus.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeReplay.hpp"

using namespace mneme;
using namespace llvm;

#ifdef MNEME_ENABLE_HIP
#include "mneme/MnemeMemoryHIP.hpp"
#include "mneme/MnemeRecordHIP.hpp"
using MnemeRecorderDevice = MnemeRecorderHIP;
using DeviceVendorTraits = DeviceTraits<DeviceVendors::HIP>;
constexpr DeviceVendors Vendor = DeviceVendors::HIP;
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
    cl::desc("The optimization level to use when optimizing IR"), cl::init('3'),
    cl::cat(MnemeCategory));

static cl::opt<unsigned> MnemeBackendOptLevel(
    "backend-opt-level",
    cl::desc("The optimization level to use when optimizing IR"), cl::init(3),
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

int main(int argc, char *argv[]) {
  cl::HideUnrelatedOptions(MnemeCategory);
  cl::ParseCommandLineOptions(argc, argv, "GPU Replay Tool\n");
  mneme::InitLLVM(argc, argv);
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
  auto Mod = proteus::linkModules(Ctx, Modules);
  proteus::pruneIR(*Mod);
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
      *Mod, *ReplayKernelFunc, RecordedGrid.x * RecordedGrid.y * RecordedGrid.z,
      RecordedBlock.x * RecordedBlock.y * RecordedBlock.z);
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
  for (int i = 0; i < _MnemeRepeats; i++) {
    LOG_DEBUG("Run {}/{}", i + 1, _MnemeRepeats);
    DeviceVendorTraits::DeviceStream_t ReplayStream;
    auto EC = DeviceVendorTraits::DeviceErrorCheck(
        DeviceVendorTraits::DeviceStreamCreate(&ReplayStream));
    if (EC)
      FATAL_ERROR("Error when creating a stream for replay\n" + EC.value());

    auto Args = RInstance.getArgs();
    EC = DeviceVendorTraits::DeviceErrorCheck(
        DeviceVendorTraits::launchKernelFunction(
            Func, RecordedGrid, RecordedBlock, Args.get(),
            RInstance.getSharedMemSize(), ReplayStream));
    if (EC)
      FATAL_ERROR("Error When Launching Kernel: " + EC.value());

    EC = DeviceVendorTraits::DeviceErrorCheck(
        DeviceVendorTraits::DeviceStreamSynchronize(ReplayStream));

    if (EC)
      FATAL_ERROR("Error When synchronizing with kernel stream: " + EC.value());

    verify &= RInstance.isMemorySame();
    RInstance.reset();
  }
  if (verify)
    std::cout << "Results Match" << "\n";
  else
    std::cout << "Results DO NOT Match" << "\n";

  RInstance.releaseMemory();
  return !verify;
}
