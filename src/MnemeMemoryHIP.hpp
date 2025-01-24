#pragma once
#include <hip/hip_runtime.h>

#include <cstdint>

#include "DeviceTraits.hpp"
#include "Logger.hpp"
#include "MnemeMemory.hpp"
#include "Utils.hpp"

namespace mneme {

class MnemeMemoryBlobHIP;

class MnemeMemoryBlobHIP : public MnemeMemoryBlob<MnemeMemoryBlobHIP, HIP> {
private:
  static inline uint64_t
  getPageSize(int DeviceID,
              const hipMemAllocationGranularity_flags Granularity) {
    uint64_t PageSize;
    hipMemAllocationProp Prop = {};
    Prop.type = hipMemAllocationTypePinned;
    Prop.location.type = hipMemLocationTypeDevice;
    Prop.location.id = DeviceID;
    // TODO: I could not find any documentation regarding the compressionType in
    // HIP. I will leave unitialized a.t.m.
    // Prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;

    hipErrCheck(hipMemGetAllocationGranularity(&PageSize, &Prop, Granularity));
    return PageSize;
  }

public:
  using MnemeMemoryBlob<MnemeMemoryBlobHIP, HIP>::MnemeMemoryBlob;

  static constexpr hipMemcpyKind MemcpyHostToDeviceKind() {
    return hipMemcpyHostToDevice;
  }

  static hipError_t DeviceCopy(void *Dest, void *Src, size_t SizeBytes,
                               hipMemcpyKind Kind) {
    return hipMemcpy(Dest, Src, SizeBytes, Kind);
  }

  void mmap(hipMemGenericAllocationHandle_t &MHandle, void *Addr,
            uintptr_t Size, int DeviceId) {
    hipMemAllocationProp Prop = {};
    Prop.type = hipMemAllocationTypePinned;
    Prop.location.type = hipMemLocationTypeDevice;
    Prop.location.id = DeviceId;
    DBG(Logger::logs("mneme") << "Requesting address with Size " << Size
                              << " on device " << DeviceID << "\n");

    hipErrCheck(hipMemCreate(&MHandle, Size, &Prop, 0));
    hipErrCheck(hipMemMap((void *)Addr, Size, 0, MHandle, 0));

    hipMemAccessDesc ADesc = {};
    ADesc.location.type = hipMemLocationTypeDevice;
    ADesc.location.id = DeviceId;
    ADesc.flags = hipMemAccessFlagsProtReadWrite;

    // Sets address
    DBG(Logger::logs("mneme")
        << "Setting Access 'RW' to " << Addr << " with size " << Size << "\n");
    hipErrCheck(hipMemSetAccess(Addr, Size, &ADesc, 1));
  }

  static uint64_t getMinPageSize(int DeviceID) {
    return getPageSize(DeviceID, hipMemAllocationGranularityMinimum);
  }

  static void *getVirtualAddress(uint64_t Size, uintptr_t VA,
                                 uint64_t Alignment) {
    hipDeviceptr_t devPtr = 0;

    hipErrCheck(hipMemAddressReserve(&devPtr, Size, Alignment,
                                     reinterpret_cast<hipDeviceptr_t>(VA), 0));
    DBG(Logger::logs("mneme") << "Allocated VASize " << Size << " at Address "
                              << std::hex << devPtr << std::dec << "\n");
    return (void *)devPtr;
  }

  void unmap(hipMemGenericAllocationHandle_t &MHandle, uintptr_t Addr,
             uintptr_t Size) {
    hipErrCheck(hipMemUnmap((void *)Addr, Size));
    hipErrCheck(hipMemRelease(MHandle));
    hipErrCheck(hipMemAddressFree((void *)Addr, Size));
  }
};

} // namespace mneme
