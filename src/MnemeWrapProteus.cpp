#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
// Use the *exact* prototype of the real symbol from Proteus.
// Replace the signature below with the real one from the header.
extern "C" int __real___jit_launch_kernel(void *Kernel, dim3 GridDim,
                                          dim3 BlockDim, void **KernelArgs,
                                          uint64_t ShmemSize,
                                          hipStream_t *Stream);

extern "C" int __wrap___jit_launch_kernel(void *Kernel, dim3 GridDim,
                                          dim3 BlockDim, void **KernelArgs,
                                          uint64_t ShmemSize,
                                          hipStream_t Stream) {
  // pre-hook
  // e.g., log arguments, modify, track timing, etc.
  printf("I wrapped kernel launch\n");
  int rc =
      hipLaunchKernel(Kernel, GridDim, BlockDim, KernelArgs, ShmemSize, Stream);

  // post-hook
  return rc;
}
