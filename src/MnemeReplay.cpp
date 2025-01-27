#include "MnemeJITProteus.hpp"
#include "MnemeReplay.hpp"
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
    cl::Required);

static cl::opt<std::string>
    MnemeKernelHash("mneme-replay-hash",
                    cl::desc("The Kernel Hash of the Recorded kernels"),
                    cl::Required);

int main(int argc, char *argv[]) {
  cl::HideUnrelatedOptions(MnemeCategory);
  cl::ParseCommandLineOptions(argc, argv, "GPU Replay Tool\n");

  ProteusJIT::InitLLVM();

  std::cout << "Kernel JSON File is " << MnemeJson
            << " with Dynamic Hash value " << MnemeKernelHash << "\n";

  auto Arch = DeviceVendorTraits::GetDeviceArch();
  Logger::logs("mneme") << "Device Architecture is " << Arch << "\n";

  ReplayInstance<Vendor> RInstance(MnemeJson, MnemeKernelHash);
  llvm::LLVMContext Ctx;
  auto Modules = RInstance.loadModules(Ctx);
  auto Mod = ProteusJIT::linkJitModule(Ctx, Modules);
  auto ReplayKernelFunc = Mod->getFunction(RInstance.getKernelName());
  internalizeModule(*Mod, [&ReplayKernelFunc](const GlobalValue &GV) {
    // Do not internalize the kernel function.
    if (&GV == ReplayKernelFunc)
      return true;

    // Internalize everything else.
    return false;
  });

  ProteusJIT::optimizeIR(*Mod, Arch);
  auto RecordedGrid = RInstance.getRecordedGrid();
  auto RecordedBlock = RInstance.getRecordedBlock();
  ProteusJIT::setLaunchBoundsForKernel(
      *Mod, *ReplayKernelFunc, RecordedGrid.x * RecordedGrid.y * RecordedGrid.z,
      RecordedBlock.x * RecordedBlock.y * RecordedBlock.z);
  ProteusJIT::runCleanupPassPipeline(*Mod);
  SmallPtrSet<void *, 8> GlobalLinkedBinaries;
  auto DeviceObject =
      ProteusJIT::codegenObject(*Mod, Arch, GlobalLinkedBinaries);
  auto VendorModule = DeviceVendorTraits::getDeviceModuleFromImage(
      DeviceObject->getBufferStart());

  RInstance.initializeDeviceMemory();
  RInstance.initializeGlobals(VendorModule);

  auto Func = DeviceVendorTraits::getKernelFunctionFromImage(
      VendorModule, RInstance.getKernelName());

  RInstance.releaseMemory();
}
