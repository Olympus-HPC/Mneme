#include <cstdint>
#if defined(_MSC_VER)
#define HAVE_DECLSPEC_DLL
#endif

#if defined(HAVE_DECLSPEC_DLL)
#define API_EXPORT(RTYPE) __declspec(dllexport) RTYPE
#else
#define API_EXPORT(RTYPE) RTYPE
#endif

extern "C" {
API_EXPORT(uint64_t) MnemePy_startProfile(const char *KernelName) {
  return 0;
}

API_EXPORT(void)
MnemePy_stopProfile(uint64_t Token, uint64_t *ProfileData, uint64_t Size) {
  // FIXME: Here I am just emiting 0 to allow the rest of the infrastructure to work properly 
  for (int i = 0; i < Size; i++)
    ProfileData[i] = 0;
  return;
}

API_EXPORT(uint64_t) MnemePy_getNumRecords(uint64_t token) {
  return 0;
}

API_EXPORT(void) MnemePy_initProfiler() {
}
}
