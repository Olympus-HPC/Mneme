#pragma once
#include <optional>
#include <stdexcept>

enum DeviceVendors { HIP, CUDA };

template <DeviceVendors Type>
struct DeviceTraits;


#ifdef MNEME_EXAMPLE_ENABLE_HIP
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/hip_runtime.h>

template <>
struct DeviceTraits<DeviceVendors::HIP> {
  using DeviceError_t = hipError_t;
  using DeviceStream_t = hipStream_t;
  using DevicePtr_t = hipDeviceptr_t;
  using DeviceHandle_t = hipDevice_t;
  using DeviceEvent_t = hipEvent_t;
  static constexpr auto DeviceSuccess = hipSuccess;
  static inline std::optional<std::string> deviceErrorCheck(
      hipError_t ErrorCode)
  {
    if (ErrorCode == hipSuccess) return std::nullopt;
    return std::string(hipGetErrorString(ErrorCode));
  }

  static hipError_t deviceStreamSynchronize(hipStream_t Stream)
  {
    return hipStreamSynchronize(Stream);
  }

  static hipError_t deviceMemset(void *DevPtr, int Value, size_t Bytes)
  {
    auto EC = hipMemset(DevPtr, Value, Bytes);
    return EC;
  }

  static hipError_t deviceMalloc(void **ptr, size_t size)
  {
    return hipMalloc(ptr, size);
  }

  template <typename T>
  static hipError_t deviceFree(T *ptr) { return hipFree(ptr); }

  template<typename T, typename P>
  static hipError_t deviceCopy(T *Dest,
                               P *Src,
                               size_t SizeBytes,
                               hipMemcpyKind Kind)
  {
    return hipMemcpy(Dest, Src, SizeBytes, Kind);
  }

  static hipError_t deviceSynchronize() { return hipDeviceSynchronize(); }


  static hipError_t getDeviceCount(int &devCount)
  {
    return hipGetDeviceCount(&devCount);
  }

  static hipError_t setDevice(int DeviceId) { return hipSetDevice(DeviceId); }

  static hipError_t getDevice(int &DeviceId) { return hipGetDevice(&DeviceId); }

  static constexpr hipMemcpyKind memcpyHostToDeviceKind()
  {
    return hipMemcpyHostToDevice;
  }

  static constexpr hipMemcpyKind memcpyDeviceToHostKind()
  {
    return hipMemcpyDeviceToHost;
  }

  static DeviceError_t deviceStreamCreate(DeviceStream_t *Stream)
  {
    return hipStreamCreate(Stream);
  }

  static DeviceError_t deviceStreamDestroy(DeviceStream_t Stream)
  {
    return hipStreamDestroy(Stream);
  }

  static DeviceError_t deviceEventCreate(DeviceEvent_t *event)
  {
    return hipEventCreate(event);
  }

  static DeviceError_t deviceEventRecord(DeviceEvent_t event,
                                         DeviceStream_t stream)
  {
    return hipEventRecord(event, stream);
  }

  static DeviceError_t deviceEventDestroy(DeviceEvent_t event)
  {
    return hipEventDestroy(event);
  }

  static DeviceError_t deviceEventSynchronize(DeviceEvent_t event)
  {
    return hipEventSynchronize(event);
  }

  static DeviceError_t deviceEventElapsedTime(float *ms,
                                              DeviceEvent_t start,
                                              DeviceEvent_t stop)
  {
    return hipEventElapsedTime(ms, start, stop);
  }

  static DeviceError_t deviceGetSymbolAddress(void **devPtr, const void *symbol)
  {
    return hipGetSymbolAddress(devPtr, symbol);
  }
};
#elif defined(MNEME_EXAMPLE_ENABLE_CUDA)

#include <cuda.h>
#include <cuda_runtime.h>

template <>
struct DeviceTraits<DeviceVendors::CUDA> {

  using DeviceError_t = cudaError_t;
  using DeviceDriverError_t = CUresult;

  using DeviceStream_t = cudaStream_t;


  using DeviceHandle_t = CUdevice;

  using DeviceEvent_t = cudaEvent_t;
  static constexpr auto DeviceSuccess = cudaSuccess;
  static constexpr auto DeviceDriverSuccess = CUDA_SUCCESS;

  static inline std::optional<std::string> deviceErrorCheck(
      DeviceError_t ErrorCode)
  {
    if (ErrorCode == DeviceSuccess) return std::nullopt;
    return std::string(cudaGetErrorString(ErrorCode));
  }

  static inline std::optional<std::string> deviceErrorCheck(
      DeviceDriverError_t ErrorCode)
  {
    if (ErrorCode == DeviceDriverSuccess) return std::nullopt;

    if (ErrorCode == CUDA_ERROR_DEINITIALIZED) {
      return std::nullopt;
    }

    const char *name = nullptr, *desc = nullptr;
    cuGetErrorName(ErrorCode, &name);
    cuGetErrorString(ErrorCode, &desc);
    auto EC = std::string("Error:") + std::to_string(ErrorCode) + ":";
    if (name) EC += std::string(name);
    if (desc) EC += std::string(" description:") + std::string(desc);

    return EC;
  }

  static DeviceError_t deviceStreamSynchronize(DeviceStream_t Stream)
  {
    return cudaStreamSynchronize(Stream);
  }

  static DeviceError_t deviceMemset(void *DevPtr, int Value, size_t Bytes)
  {
    auto EC = cudaMemset(DevPtr, Value, Bytes);
    return EC;
  }

  static DeviceError_t deviceMalloc(void **ptr, size_t size)
  {
    return cudaMalloc(ptr, size);
  }

  static DeviceError_t deviceFree(void *ptr) { return cudaFree(ptr); }

  template<typename T, typename P>
  static DeviceError_t deviceCopy(T *Dest,
                                  P *Src,
                                  size_t SizeBytes,
                                  cudaMemcpyKind Kind)
  {
    return cudaMemcpy(Dest, Src, SizeBytes, Kind);
  }

  static DeviceError_t deviceSynchronize() { return cudaDeviceSynchronize(); }

  static cudaError_t getDeviceCount(int &devCount)
  {
    return cudaGetDeviceCount(&devCount);
  }

  static cudaError_t setDevice(int DeviceId) { return cudaSetDevice(DeviceId); }

  static cudaError_t getDevice(int &DeviceId)
  {
    return cudaGetDevice(&DeviceId);
  }

  static constexpr cudaMemcpyKind memcpyHostToDeviceKind()
  {
    return cudaMemcpyHostToDevice;
  }

  static constexpr cudaMemcpyKind memcpyDeviceToHostKind()
  {
    return cudaMemcpyDeviceToHost;
  }

  static DeviceError_t deviceStreamCreate(DeviceStream_t *Stream)
  {
    return cudaStreamCreate(Stream);
  }

  static DeviceError_t deviceStreamDestroy(DeviceStream_t Stream)
  {
    return cudaStreamDestroy(Stream);
  }

  static DeviceError_t deviceEventCreate(DeviceEvent_t *event)
  {
    return cudaEventCreate(event);
  }

  static DeviceError_t deviceEventRecord(DeviceEvent_t event,
                                         DeviceStream_t stream)
  {
    return cudaEventRecord(event, stream);
  }

  static DeviceError_t deviceEventDestroy(DeviceEvent_t event)
  {
    return cudaEventDestroy(event);
  }

  static DeviceError_t deviceEventSynchronize(DeviceEvent_t event)
  {
    return cudaEventSynchronize(event);
  }

  static DeviceError_t deviceEventElapsedTime(float *ms,
                                              DeviceEvent_t start,
                                              DeviceEvent_t stop)
  {
    return cudaEventElapsedTime(ms, start, stop);
  }

  static DeviceError_t deviceGetSymbolAddress(void **devPtr, const void *symbol)
  {
    return cudaGetSymbolAddress(devPtr, symbol);
  }
};
#endif

#ifdef MNEME_EXAMPLE_ENABLE_HIP
using Device = DeviceTraits<DeviceVendors::HIP>;
#elif defined(MNEME_EXAMPLE_ENABLE_CUDA)
using Device = DeviceTraits<DeviceVendors::CUDA>;
#endif

template <class ErrT, class Traits>
inline void check(ErrT ec, const char *file, int line)
{
  if (auto msg = Traits::deviceErrorCheck(ec)) {
    fprintf(stderr, "ERROR @ %s:%d -> %s\n", file, line, msg->c_str());
    std::abort();
  }
}

#define DEVICE_CHECK(CALL) \
  check<decltype(CALL), Device>((CALL), __FILE__, __LINE__)
