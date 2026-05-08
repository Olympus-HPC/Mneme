#include "MnemeAnnotationRuntime.hpp"
#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeLLVMUtils.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeRecord.hpp"
#include <hip/hip_runtime.h>
#include <utility>
#include <csignal>
#include <cstdlib>
#include <unistd.h>
#include <atomic>

using namespace mneme;


static void signal_handler(int signal) {
  static std::atomic_flag in_handler = ATOMIC_FLAG_INIT;
  if (in_handler.test_and_set()) _exit(128 + signal);

  // Only use async-signal-safe functions in signal handlers
  // spdlog is NOT safe to use here, especially during program exit
  const char msg[] = "=== PROGRAM CRASHED (see core dump) ===\n";
  write(STDERR_FILENO, msg, sizeof(msg) - 1);

  // SA_RESETHAND already restored SIG_DFL; just re-raise for the core dump.
  std::raise(signal);
}

static void install_crash_handler() {
  struct sigaction sa;
  sa.sa_handler = signal_handler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_RESETHAND;
  for (int sig : {SIGSEGV, SIGABRT, SIGFPE, SIGILL, SIGBUS}) {
    sigaction(sig, &sa, nullptr);
  }
}

class MnemeRecorderHIPPreload
    : public MnemeRecorder<mneme::DeviceVendors::HIP> {
private:
  static constexpr bool hasFatBinEnd = false;
  MnemeRecorderHIPPreload(MnemeRecorderHIPPreload &) = delete;
  MnemeRecorderHIPPreload(MnemeRecorderHIPPreload &&) = delete;

  MnemeRecorderHIPPreload() {
    // install crash handler on first HIP call (after main has started)
    install_crash_handler();
  }

public:
  static MnemeRecorderHIPPreload &instance() {
    static MnemeRecorderHIPPreload Recorder{};
    return Recorder;
  }
};

extern "C" {
hipError_t hipMalloc(void **ptr, size_t size) {
  LOG_DEBUG("Entering Mneme to Malloc pointer of size : {}", size);
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.rtMalloc(ptr, size);
}

hipError_t hipMallocManaged(void **ptr, size_t size, unsigned int flags) {
  LOG_DEBUG("Entering Mneme to Malloc Managed pointer of size : {}", size);
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.rtManagedMalloc(ptr, size, flags);
};

hipError_t hipHostMalloc(void **ptr, size_t size, unsigned int flags) {
  LOG_DEBUG("Entering Mneme to Malloc 'Host|Pinned' pointer of size : {}",
            size);
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.rtHostMalloc(ptr, size, flags);
}

hipError_t hipFree(void *ptr) {
  LOG_DEBUG("Entering Mneme to Free pointer");
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.rtFree(ptr);
};

hipError_t hipHostFree(void *ptr) {
  LOG_DEBUG("Entering Mneme to HostFree pointer");
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.rtHostFree(ptr);
}

hipError_t hipSetDevice(int deviceID) {
  LOG_DEBUG("Entering Mneme to set Device");
  auto &mneme = MnemeRecorderHIPPreload::instance();
  auto result = mneme.rtSetDevice(deviceID);
  return result;
}

hipError_t hipGetDevice(int *deviceID) {
  LOG_DEBUG("Entering Mneme to set Device");
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.rtGetDevice(deviceID);
}

hipError_t __proteus_launch_kernel(void *Kernel, dim3 GridDim, dim3 BlockDim,
                               void **KernelArgs, uint64_t ShmemSize,
                               void *Stream) {
  LOG_DEBUG("Enetering Mneme to launch kernel");
  auto &mneme = MnemeRecorderHIPPreload::instance();
  auto result = mneme.rtLaunchKernel(Kernel, GridDim, BlockDim, KernelArgs, ShmemSize,
                                     static_cast<hipStream_t>(Stream));

  return result;
}

bool mneme_set_metadata_for_ptr(const void *ptr, mneme::Metadata md) {
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.setMetadataForPointer(ptr, std::move(md));
}

bool mneme_get_metadata_for_ptr(const void *ptr, mneme::Metadata *md) {
  if (!md)
    return false;

  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.getMetadataForPointer(ptr, *md);
}

bool mneme_erase_metadata_for_ptr(const void *ptr) {
  auto &mneme = MnemeRecorderHIPPreload::instance();
  return mneme.eraseMetadataForPointer(ptr);
}
}
