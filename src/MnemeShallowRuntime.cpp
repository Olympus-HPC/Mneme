// This acts as a NOP library. That adds minimal overhead when device binaries
// are initialized. It provides hooks to "runtimes" to intercept and pull the
// LLVM IR and associate it appropriately with all binaries/linages.
#include <cstdint>

#ifdef MNEME_ENABLE_CUDA
#include <cuda_runtime.h>
#elif defined(MNEME_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

extern "C" {

void __register_linked_binary(void *FatbinWrapper, const char *ModuleId) {}

// Only required by CUDA.
void __register_fatbinary_end(void **) {}

void __register_fatbinary(void **Handle, void *FatbinWrapper,
                          const char *ModuleId) {}

void __register_var(void **Handle, char *HostVar, char *DeviceAddress,
                    const char *DeviceName, int Ext, std::size_t Size,
                    int Constant, int Global) {}

void __register_function(void **Handle, const char *HostFun, char *DeviceFun,
                         const char *DeviceName, int ThreadLimit, uint3 *tid,
                         uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize) {}
}
