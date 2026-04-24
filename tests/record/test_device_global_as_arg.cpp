// clang-format off
// RUN: rm -rf *.json Recorded*.bc DeviceState*.mneme
// RUN: LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_LOG_LEVEL=debug MNEME_PAGE_SIZE=%PG ./test_device_global_as_arg%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: %RR | %FILECHECK %s --check-prefix=CHECK-RR
// RUN: rm -rf *.json Recorded*.bc DeviceState*.mneme
// clang-format on
//
// Minimum reproducer: a __device__ global lives in a *different* TU (device
// module) than the kernel. The other TU's host helper fetches the global's
// device address via deviceGetSymbolAddress and passes it as a raw pointer
// kernel arg.
//
// In non-RDC mode each TU becomes its own device binary, so the kernel's
// module does not contain `g_data`. Proteus's per-binary global mapping for
// this kernel therefore does NOT include `g_data`, and the pointer was never
// routed through the Mneme malloc interceptor, so AllocatedBlobs has nothing
// for it either. Yet the 8 raw pointer bytes still land verbatim in Args,
// and on replay those bytes are reconstructed into argv with no
// global/blob state to back them — the replayed kernel dereferences a
// stale device address.

#include <cstdio>
#include <iostream>

#include "mneme/DeviceTraits.hpp"
using namespace mneme;

#ifdef MNEME_ENABLE_HIP
using MnemeDeviceRT = DeviceTraits<DeviceVendors::HIP>;
#elif defined(MNEME_ENABLE_CUDA)
using MnemeDeviceRT = DeviceTraits<DeviceVendors::CUDA>;
#endif

static constexpr int N = 16;

// Declared/defined in the other TU (test_device_global_as_arg_global.cpp).
int *get_foreign_global_addr();
void copy_foreign_global_out(int *host_buf);

// Kernel references only its `int*` argument. The foreign `g_data` symbol
// does not appear in this TU's device module.
__global__ void write_through_raw_ptr(int *p, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n)
    p[idx] = idx * idx;
}

int main() {
  int *sym_addr = get_foreign_global_addr();
  if (!sym_addr) {
    std::cout << "GetSymbolAddress failed\n";
    return -1;
  }

  write_through_raw_ptr<<<1, N>>>(sym_addr, N);

  auto EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceSynchronize());
  if (EC) {
    std::cout << "Device error: " << EC.value() << "\n";
    return -1;
  }

  int h_data[N] = {0};
  copy_foreign_global_out(h_data);
  for (int i = 0; i < N; i++) {
    if (h_data[i] != i * i) {
      std::cout << "Mismatch at " << i << ": got " << h_data[i] << " expected "
                << (i * i) << "\n";
      return -1;
    }
  }

  std::cout << "OK\n";
  return 0;
}

// clang-format off
// CHECK: OK
// CHECK-RR: DemangledName: write_through_raw_ptr(int*, int)
// CHECK-RR: NumModules: 2
// CHECK-RR: NumInstances: 1
// CHECK-RR: CapturedGlobals: 1 g_data
// CHECK-RR: PointerOffsets: 1
// clang-format on
