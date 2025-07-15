#ifndef UTILS_H
#define UTILS_H


#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <climits>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <cuda.h>

constexpr int const BLOCK_SIZE = 256;

// only used for min/max
constexpr int const BLOCK_WIDTH = 4096;

#define CUDA_RT_CALL(call) call

#ifdef DEBUG
#undef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                     \
  {                                                                            \
    cudaError_t cudaStatus = call;                                             \
    if (cudaSuccess != cudaStatus) {                                           \
      fprintf(                                                                 \
          stderr,                                                              \
          "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with %s "   \
          "(%d).\n",                                                           \
          #call,                                                               \
          __LINE__,                                                            \
          __FILE__,                                                            \
          cudaGetErrorString(cudaStatus),                                      \
          cudaStatus);                                                         \
      abort();                                                                 \
    }                                                                          \
  }
#endif

#define NVCOMP_TYPE_SWITCH(type_var, func, ...)                                \
  do {                                                                         \
    switch (type_var) {                                                        \
    case NVCOMP_TYPE_CHAR:                                                     \
      func<char, uint32_t, char>(__VA_ARGS__);                                 \
      break;                                                                   \
    case NVCOMP_TYPE_UCHAR:                                                    \
      func<unsigned char, uint32_t, unsigned char>(__VA_ARGS__);               \
      break;                                                                   \
    case NVCOMP_TYPE_SHORT:                                                    \
      func<short, uint32_t, short>(__VA_ARGS__);                               \
      break;                                                                   \
    case NVCOMP_TYPE_USHORT:                                                   \
      func<unsigned short, uint32_t, unsigned short>(__VA_ARGS__);             \
      break;                                                                   \
    case NVCOMP_TYPE_INT:                                                      \
      func<int, uint32_t, int>(__VA_ARGS__);                                   \
      break;                                                                   \
    case NVCOMP_TYPE_UINT:                                                     \
      func<unsigned int, uint32_t, unsigned int>(__VA_ARGS__);                 \
      break;                                                                   \
    case NVCOMP_TYPE_LONGLONG:                                                 \
      func<long long, uint64_t, long long>(__VA_ARGS__);                       \
      break;                                                                   \
    case NVCOMP_TYPE_ULONGLONG:                                                \
      func<unsigned long long, uint64_t, unsigned long long>(__VA_ARGS__);     \
      break;                                                                   \
    default:                                                                   \
      throw std::runtime_error("Unknown type: " + std::to_string(type_var));   \
    }                                                                          \
  } while (0)


/* Supported datatypes */
typedef enum nvcompType_t
{
  NVCOMP_TYPE_CHAR = 0,      // 1B
  NVCOMP_TYPE_UCHAR = 1,     // 1B
  NVCOMP_TYPE_SHORT = 2,     // 2B
  NVCOMP_TYPE_USHORT = 3,    // 2B
  NVCOMP_TYPE_INT = 4,       // 4B
  NVCOMP_TYPE_UINT = 5,      // 4B
  NVCOMP_TYPE_LONGLONG = 6,  // 8B
  NVCOMP_TYPE_ULONGLONG = 7, // 8B
  NVCOMP_TYPE_BITS = 0xff    // 1b
} nvcompType_t;/* Supported datatypes */


template <typename T>
T unpackBytes(
    const void* data, const uint8_t numBits, const T minValue, const size_t i)
{
  using U = typename std::make_unsigned<T>::type;

  if (numBits == 0) {
    return minValue;
  } else {
    // enough space to hold 64 bits with up to 7 bit offset
    uint8_t scratch[9] = {0,0,0,0,0,0,0,0,0};

    // shifting by width of the type is UB
    const U mask = numBits < sizeof(T)*8U ? static_cast<U>((1ULL << numBits) -
        1) : static_cast<U>(-1);
    const uint8_t* byte_data = reinterpret_cast<decltype(byte_data)>(data);

    // Specialized
    // Need to copy into scratch because
    // GPU can only address n byte datatype on multiple of n address
    // TODO: add an optimized version in case numBits aligns to word size
    //       boundaries (1,2,4,8,16,32 and 64 bits)
    size_t start_byte = (i * numBits) / 8;
    // end_byte needed so we don't attempt to read from illegal memory
    size_t end_byte = ((i + 1) * numBits - 1) / 8;
    assert(end_byte - start_byte <= sizeof(scratch));

    for (size_t j = start_byte, k = 0; j <= end_byte; ++j, ++k) {
      scratch[k] = byte_data[j];
    }

    const int bitOffset = (i * numBits) % 8;
    U baseValue = 0;
    for (size_t k = 0; k <= end_byte - start_byte; ++k) {
      U shifted;
      if (k > 0) {
        shifted = static_cast<U>(scratch[k]) << ((k * 8) - bitOffset);
      } else {
        shifted = static_cast<U>(scratch[k]) >> bitOffset;
      }
      baseValue |= mask & shifted;
    }

    const T value = baseValue + minValue;
    return value;
  }
}

size_t getReduceScratchSpaceSize(size_t const num);
size_t requiredWorkspaceSize(size_t const num, const nvcompType_t type);

// returns nano-seconds
inline uint64_t get_time(timespec start, timespec end)
{
  constexpr const uint64_t BILLION = 1000000000ULL;
  const uint64_t elapsed_time
      = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
  return elapsed_time;
}

// size in bytes, returns GB/s
inline double gibs(struct timespec start, struct timespec end, size_t s)
{
  uint64_t t = get_time(start, end);
  return (double)s / t * 1e9 / 1024 / 1024 / 1024;
}

// size in bytes, returns GB/s
inline double
gbs(const std::chrono::time_point<std::chrono::steady_clock>& start,
    const std::chrono::time_point<std::chrono::steady_clock>& end,
    size_t s)
{
  return (double)s / std::chrono::nanoseconds(end - start).count();
}

template <typename T>
T* align(T* const ptr, const size_t alignment)
{
  const size_t bits = reinterpret_cast<size_t>(ptr);
  const size_t mask = alignment - 1;

  return reinterpret_cast<T*>(((bits - 1) | mask) + 1);
}

template <typename T>
size_t
relativeEndOffset(const void* start, const T* subsection, const size_t length)
{
  std::ptrdiff_t diff = reinterpret_cast<const char*>(subsection)
                        - static_cast<const char*>(start);
  return static_cast<size_t>(diff) + length * sizeof(T);
}

template <typename T = size_t>
T relativeEndOffset(const void* start, const void* subsection)
{
  std::ptrdiff_t diff = reinterpret_cast<const char*>(subsection)
                        - static_cast<const char*>(start);
  return static_cast<T>(diff);
}

template <typename U, typename T>
constexpr __host__ __device__ U roundUpDiv(U const num, T const chunk)
{
  return (num / chunk) + (num % chunk > 0);
}

template <typename U, typename T>
constexpr __host__ __device__ U roundDownTo(U const num, T const chunk)
{
  return (num / chunk) * chunk;
}

template <typename U, typename T>
constexpr __host__ __device__ U roundUpTo(U const num, T const chunk)
{
  return roundUpDiv(num, chunk) * chunk;
}

template <typename T>
inline nvcompType_t TypeOf()
{
  if (std::is_same<T, int8_t>::value) {
    return NVCOMP_TYPE_CHAR;
  } else if (std::is_same<T, uint8_t>::value) {
    return NVCOMP_TYPE_UCHAR;
  } else if (std::is_same<T, int16_t>::value) {
    return NVCOMP_TYPE_SHORT;
  } else if (std::is_same<T, uint16_t>::value) {
    return NVCOMP_TYPE_USHORT;
  } else if (std::is_same<T, int32_t>::value) {
    return NVCOMP_TYPE_INT;
  } else if (std::is_same<T, uint32_t>::value) {
    return NVCOMP_TYPE_UINT;
  } else if (std::is_same<T, int64_t>::value) {
    return NVCOMP_TYPE_LONGLONG;
  } else if (std::is_same<T, uint64_t>::value) {
    return NVCOMP_TYPE_ULONGLONG;
  } else {
    return NVCOMP_TYPE_INT;
  }
}

__inline__ size_t sizeOfnvcompType(nvcompType_t type)
{
  switch (type) {
  case NVCOMP_TYPE_BITS:
    return 1;
  case NVCOMP_TYPE_CHAR:
    return sizeof(int8_t);
  case NVCOMP_TYPE_UCHAR:
    return sizeof(uint8_t);
  case NVCOMP_TYPE_SHORT:
    return sizeof(int16_t);
  case NVCOMP_TYPE_USHORT:
    return sizeof(uint16_t);
  case NVCOMP_TYPE_INT:
    return sizeof(int32_t);
  case NVCOMP_TYPE_UINT:
    return sizeof(uint32_t);
  case NVCOMP_TYPE_LONGLONG:
    return sizeof(int64_t);
  case NVCOMP_TYPE_ULONGLONG:
    return sizeof(uint64_t);
  default:
    throw std::runtime_error("Unsupported type " + std::to_string(type));
  }
}

void compress(
    void* const workspace,
    const size_t workspaceSize,
    const nvcompType_t inType,
    void* const outPtr,
    const void* const in,
    const size_t* const numPtr,
    const size_t maxNum,
    void* const minValueDevicePtr,
    unsigned char* const numBitsDevicePtr);

#endif
