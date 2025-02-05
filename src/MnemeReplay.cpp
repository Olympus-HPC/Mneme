#include "MnemeJITProteus.hpp"
#include "MnemeLogger.hpp"
#include "MnemeReplay.hpp"
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/CommandLine.h>

using namespace mneme;
using namespace llvm;

#include "DeviceTraits.hpp"
#ifdef ENABLE_HIP
#include "MnemeMemoryHIP.hpp"
#include "MnemeRecordHIP.hpp"
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

static cl::opt<int>
    MnemeOptLevel("opt-level",
                  cl::desc("The optimization level to use when optimizing IR"),
                  cl::init(3), cl::cat(MnemeCategory));
static cl::alias MnemeShortOptLevel(
    "O", cl::desc("The optimization level to use when optimizing IR"),
    cl::aliasopt(MnemeOptLevel), cl::cat(MnemeCategory));
static cl::opt<int>
    MnemeRepeats("repeats",
                 cl::desc("The number of repeat executions for every kernel"),
                 cl::init(3), cl::cat(MnemeCategory));

int main(int argc, char *argv[]) {
  cl::HideUnrelatedOptions(MnemeCategory);
  cl::ParseCommandLineOptions(argc, argv, "GPU Replay Tool\n");

  ProteusJIT::InitLLVM();

  LOG_INFO("using mneme db-file {} and dynamic instance {}", MnemeJson,
           MnemeKernelHash);
  auto Arch = DeviceVendorTraits::GetDeviceArch();
  LOG_INFO("Detected system device Architecture is: {}", Arch);

  ReplayInstance<Vendor> RInstance(MnemeJson, MnemeKernelHash);
  llvm::LLVMContext Ctx;
  auto Modules = RInstance.loadModules(Ctx);
  auto Mod = ProteusJIT::linkJitModule(Ctx, Modules);
  ProteusJIT::pruneIR(*Mod);
  pruneMnemeGlobals(*Mod);
  auto ReplayKernelFunc = Mod->getFunction(RInstance.getKernelName());
  internalizeModule(*Mod, [&ReplayKernelFunc](const GlobalValue &GV) {
    if (isa<GlobalVariable>(GV))
      return true;
    // Do not internalize the kernel function.
    if (&GV == ReplayKernelFunc)
      return true;

    // Internalize everything else.
    return false;
  });

  LOG_INFO("Optimizing Kernel with OptLevel {}", MnemeOptLevel.getValue());
  ProteusJIT::optimizeIR(*Mod, Arch, MnemeOptLevel);

  auto RecordedGrid = RInstance.getRecordedGrid();
  auto RecordedBlock = RInstance.getRecordedBlock();
  ProteusJIT::setLaunchBoundsForKernel(
      *Mod, *ReplayKernelFunc, RecordedGrid.x * RecordedGrid.y * RecordedGrid.z,
      RecordedBlock.x * RecordedBlock.y * RecordedBlock.z);
  ProteusJIT::runCleanupPassPipeline(*Mod);
  SmallPtrSet<void *, 8> GlobalLinkedBinaries;
  Mod->print(llvm::outs(), nullptr);
  auto DeviceObject =
      ProteusJIT::codegenObject(*Mod, Arch, GlobalLinkedBinaries);
  auto VendorModule = DeviceVendorTraits::getDeviceModuleFromImage(
      DeviceObject->getBufferStart());

  RInstance.initializeDeviceMemory();
  RInstance.initializeGlobals(VendorModule);

  auto Func = DeviceVendorTraits::getKernelFunctionFromImage(
      VendorModule, RInstance.getKernelName());

  bool verify = true;
  for (int i = 0; i < MnemeRepeats; i++) {
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
