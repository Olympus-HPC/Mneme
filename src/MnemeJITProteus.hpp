#ifdef ENABLE_HIP
#include <JitEngineDeviceHIP.hpp>
#elif defined(ENABLE_CUDA)
#else
#error "Please define ENABLE_HIP or ENABLE_CUDA"
#endif
