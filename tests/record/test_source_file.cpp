// clang-format off
// RUN: rm -rf "%t.$$.mneme" && mkdir -p "%t.$$.mneme"
// RUN: LD_PRELOAD=MNEME_PRELOAD_LIB MNEME_LOG_LEVEL=debug MNEME_PAGE_SIZE=%PG MNEME_DATA_DIR="%t.$$.mneme" MNEME_COPY_SOURCE=1 %build/test_source_file%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: %RR "%t.$$.mneme" | %FILECHECK %s --check-prefix=CHECK-RR
// RUN: rm -rf "%t.$$.mneme"
// clang-format on

// Recording a kernel defined in its translation unit must report the absolute
// source path and, with MNEME_COPY_SOURCE, copy the file into the record dir.

#include <climits>
#include <cstdio>
#include <iostream>

#include "mneme/DeviceTraits.hpp"
using namespace mneme;

#ifdef MNEME_ENABLE_HIP
using MnemeDeviceRT = DeviceTraits<DeviceVendors::HIP>;
#elif defined(MNEME_ENABLE_CUDA)
using MnemeDeviceRT = DeviceTraits<DeviceVendors::CUDA>;
#endif

// CHECK-RR:DemangledName: source_file_kernel()
// CHECK-RR:SourceFile: {{.*}}/test_source_file.cpp
// CHECK-RR:SourceLine: {{[0-9]+}}
// CHECK-RR:SourceEndLine: {{[0-9]+}}
// CHECK-RR:SourceCopy: RecordedSource_{{[0-9a-f]+}}_test_source_file.cpp
// CHECK-RR:SourceMD5: ok
// CHECK-RR:SourceText: __global__ void source_file_kernel()
__global__ void source_file_kernel() { printf("Kernel\n"); }

int main() {
  source_file_kernel<<<1, 1>>>();
  auto EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceSynchronize());
  if (EC) {
    std::cout << "Error when running benchmark " << EC.value() << "\n";
    return -1;
  }
  return 0;
}

// CHECK: Kernel
