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
using MnemeMemoryBlobDevice = MnemeMemoryBlobHIP;
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

  std::cout << "Kernel JSON File is " << MnemeJson
            << " with Dynamic Hash value " << MnemeKernelHash << "\n";

  ReplayInstance<MnemeMemoryBlobDevice, Vendor> RInstance(MnemeJson,
                                                          MnemeKernelHash);
  llvm::LLVMContext Ctx;
  auto Mod = ProteusJIT::linkJitModule(Ctx, RInstance.getModules());
  interalize(Mod, RInstance.getKernelName());
  auto F = Mod.getFunction(RInstance.getKernelName());
  ProteusJIT::setLaunchBoundsForKernel(F);
  ProteusJIT::cleanup(M);
  ProteusJIT::optimizeIR(M);
  auto DeviceObject = ProteusJIT::codegenObject(M);
}
