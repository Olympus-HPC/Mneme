#ifdef ENABLE_HIP
#include <proteus/JitEngineDeviceHIP.hpp>
using ProteusJIT = proteus::JitEngineDeviceHIP;
#elif defined(ENABLE_CUDA)
#else
#error "Please define ENABLE_HIP or ENABLE_CUDA"
#endif
