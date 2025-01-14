#include "MnemeMemoryHIP.hpp"
#include "MnemeRecord.hpp"
#include <dlfcn.h>
#include <hip/hip_runtime.h>

namespace mneme {

class MnemeRecorderHIP
    : public MnemeRecorder<MnemeRecorderHIP, MnemeMemoryBlobHIP, HIP> {
private:
  MnemeRecorderHIP() = default;

  const std::string &getArch();

public:
  static auto *getRTLib() { return dlopen("libamdhip64.so", RTLD_NOW); }
  static const char *getLaunchKernelFnName() { return "hipLaunchKernel"; }
  static const char *getDeviceMallocFnName() { return "hipMalloc"; }
  static const char *getPinnedMallocFnName() { return "hipHostMalloc"; }
  static const char *getManagedMallocFnName() { return "hipMallocManaged"; }
  static const char *getDeviceFreeFnName() { return "hipFree"; }
  static const char *getPinnedFreeFnName() { return "hipHostFree"; }
  static const char *getUURegisterFunctionFnName() {
    return "__hipRegisterFunction";
  }
  static const char *getUURegisterVarFnName() { return "__hipRegisterVar"; }
  static const char *getUURegisterFatbinFnName() {
    return "__hipRegisterFatBinary";
  }

  static hipError_t DeviceStreamSynchronize(hipStream_t Stream) {
    return hipStreamSynchronize(Stream);
  }

  static constexpr bool hasFatBinEnd = false;

  static MnemeRecorderHIP &instance();

  void extractIR();
  void initializeGlobal(GlobalVarInfo &GVar);

  MnemeRecorderHIP(MnemeRecorderHIP &) = delete;
  MnemeRecorderHIP(MnemeRecorderHIP &&) = delete;
};
} // namespace mneme
