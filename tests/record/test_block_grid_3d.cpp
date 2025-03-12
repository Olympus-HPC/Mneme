// clang-format off
// RUN: rm -rf *.json Recorded*.bc DeviceState*.mneme
// RUN: LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_LOG_LEVEL=debug MNEME_PAGE_SIZE=%PG ./test_block_grid_3d%ext | FileCheck %s --check-prefixes=CHECK
// RUN: %RR | FileCheck %s --check-prefix=CHECK-RR
// RUN: rm -rf *.json Recorded*.bc DeviceState*.mneme
// clang-format on

#include <climits>
#include <cstdio>

#include "mneme/DeviceTraits.hpp"
using namespace mneme;

#ifdef MNEME_ENABLE_HIP
using MnemeDeviceRT = DeviceTraits<DeviceVendors::HIP>;
#endif

// CHECK-RR:DemangledName: kernel()
// CHECK-RR:NumModules: 1
// CHECK-RR:NumInstances: 2
__global__ void kernel() {
  int idx = threadIdx.z + blockIdx.z * blockDim.z;
  if (idx == gridDim.z * blockDim.z - 1) {
    printf("ThreadId: (%d %d %d) BlockID: (%d %d %d) BlockDim: (%d %d %d) "
           "GridDim: (%d %d %d)\n",
           (int)threadIdx.x, (int)threadIdx.y, (int)threadIdx.z,
           (int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z, (int)blockDim.x,
           (int)blockDim.y, (int)blockDim.z, (int)gridDim.x, (int)gridDim.y,
           (int)gridDim.z);
  }
}

int main() {
  for (int tid = 1; tid <= 2; tid++) {
    dim3 blockDim(1, 1, tid * 32);
    dim3 gridDim(1, 1, tid);
    kernel<<<gridDim, blockDim>>>();
    auto EC =
        MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceSynchronize());
    if (EC) {
      std::cout << "Error when running benchmark " << EC.value() << "\n";
      return -1;
    }
  }
  return 0;
}

// clang-format off
// CHECK:ThreadId: (0 0 31) BlockID: (0 0 0) BlockDim: (1 1 32) GridDim: (1 1 1)
// CHECK:ThreadId: (0 0 63) BlockID: (0 0 1) BlockDim: (1 1 64) GridDim: (1 1 2)
//
// CHECK-RR:BlockDims:(1, 1, 64)
// CHECK-RR:GridDims:(1, 1, 2)
// CHECK-RR:BlockDims:(1, 1, 32)
// CHECK-RR:GridDims:(1, 1, 1)

// clang-format on
