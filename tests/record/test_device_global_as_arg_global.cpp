// Companion TU for test_device_global_as_arg.cpp.
//
// Owns the __device__ global and the host helpers that expose its device
// address / host view. Lives in a separate translation unit on purpose so
// that in non-RDC builds the kernel's device module does not contain the
// `g_data` symbol.

#include "mneme/DeviceTraits.hpp"
using namespace mneme;

#ifdef MNEME_ENABLE_HIP
using MnemeDeviceRT = DeviceTraits<DeviceVendors::HIP>;
#elif defined(MNEME_ENABLE_CUDA)
using MnemeDeviceRT = DeviceTraits<DeviceVendors::CUDA>;
#endif

static constexpr int N = 16;

__device__ int g_data[N];

int *get_foreign_global_addr() {
  int *sym_addr = nullptr;
  auto EC =
      MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::deviceGetSymbolAddress(
          reinterpret_cast<void **>(&sym_addr), g_data));
  if (EC)
    return nullptr;
  return sym_addr;
}

void copy_foreign_global_out(int *host_buf) {
  int *sym_addr = get_foreign_global_addr();
  auto EC = MnemeDeviceRT::DeviceErrorCheck(
      MnemeDeviceRT::DeviceCopy(host_buf, sym_addr, N * sizeof(int),
                                MnemeDeviceRT::MemcpyDeviceToHostKind()));
  if (EC)
    return;
}
