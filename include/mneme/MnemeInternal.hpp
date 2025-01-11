#ifdef MNEME_ENABLE_HIP
#include "mneme/MnemeRecordHIP.hpp"
using MnemeRecordImplT = mneme::MnemeRecorderHIP;
#elif defined(MNEME_ENABLE_CUDA)
#error Implementation is pending
#else
#error Neither MNEME_ENABLE_HIP nor MNEME_ENABLE_CUDA is defined
#endif
