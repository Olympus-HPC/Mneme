#include "mneme/DeviceTraits.hpp"
using namespace mneme;

#ifdef MNEME_ENABLE_HIP
using MnemeDeviceRT = DeviceTraits<DeviceVendors::HIP>;
#endif

struct kernel_body_t {
  __device__ void operator()() { printf("Kernel body\n"); }
};

const kernel_body_t kernel_body{};

template <typename LB> __global__ void kernel(LB lb) { lb(); }

template <typename T>
__attribute__((always_inline)) MnemeDeviceRT::DeviceError_t launcher(T lb) {
  auto func = reinterpret_cast<const void *>(&kernel<T>);
  void *args[] = {(void *)&lb};
  return MnemeDeviceRT::deviceLaunchKernel(func, 1, 1, args, 0, 0);
}
