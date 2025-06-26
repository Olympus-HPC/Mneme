#pragma once
#include "llvm/ADT/DenseMapInfo.h"
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <llvm/ADT/Twine.h>
#include <string>

#if MNEME_ENABLE_DEBUG
#define MNEME_DBG(x) x;
#else
#define MNEME_DBG(x)
#endif

#define FATAL_ERROR(x)                                                         \
  report_fatal_error(llvm::Twine(std::string{} + __FILE__ + ":" +              \
                                 std::to_string(__LINE__) + " => " + x))

#ifdef MNEME_ENABLE_HIP
#include <hip/hip_runtime.h>

#define hipErrCheck(CALL)                                                      \
  {                                                                            \
    hipError_t err = CALL;                                                     \
    if (err != hipSuccess) {                                                   \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             hipGetErrorString(err));                                          \
      abort();                                                                 \
    }                                                                          \
  }

#define hiprtcErrCheck(CALL)                                                   \
  {                                                                            \
    hiprtcResult err = CALL;                                                   \
    if (err != HIPRTC_SUCCESS) {                                               \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             hiprtcGetErrorString(err));                                       \
      abort();                                                                 \
    }                                                                          \
  }

#elif defined(MNEME_ENABLE_CUDA)
#define cudaErrCheck(CALL)                                                     \
  {                                                                            \
    cudaError_t err = CALL;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             hipGetErrorString(err));                                          \
      abort();                                                                 \
    }                                                                          \
  }

#define cuErrCheck(CALL)                                                       \
  {                                                                            \
    CUresult err = CALL;                                                       \
    if (err != CUDA_SUCCESS) {                                                 \
      const char *name = nullptr, *desc = nullptr;                             \
      cuGetErrorName(err, &name);                                              \
      cuGetErrorString(err, &desc);                                            \
      fprintf(stderr, "CUDA Driver Error [%s]: %s\n", name ? name : "Unknown", \
              desc ? desc : "No description");                                 \
      abort();                                                                 \
    }                                                                          \
  }
#endif

namespace mneme {
namespace util {
template <typename Ty> Ty roundUp(Ty Size, Ty Divider) {
  return (Size + Divider - 1) & ~(Divider - 1);
}
template <typename Ty> Ty extractScalar(const char *&Buffer) {
  Ty Value = *reinterpret_cast<const Ty *>(Buffer);
  Buffer += sizeof(Ty);
  return Value;
}

template <typename T> std::string pointerToHexString(T *ptr) {
  std::ostringstream oss;
  oss << "0x" << std::hex << std::setw(sizeof(void *) * 2) << std::setfill('0')
      << reinterpret_cast<std::uintptr_t>(ptr);
  return oss.str();
}

template <typename T> T *hexStringToPointer(const std::string &HEXStr) {
  uint64_t Addr = std::stoul(HEXStr, nullptr, 16);
  // Cast the integer to void*
  return reinterpret_cast<T *>(Addr);
}

} // namespace util
} // namespace mneme

namespace llvm {
template <> struct DenseMapInfo<std::string> {
  static inline std::string getEmptyKey() {
    return std::string(); // Define an empty key
  }
  static inline std::string getTombstoneKey() {
    return std::string("<TOMBSTONE_KEY>");
  }
  static unsigned getHashValue(const std::string &Key) {
    // Use std::hash for hashing the string
    return std::hash<std::string>{}(Key);
  }
  static bool isEqual(const std::string &LHS, const std::string &RHS) {
    return LHS == RHS;
  }
};
} // namespace llvm
