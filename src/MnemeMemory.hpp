#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>
#include <iostream>
#include <llvm/ADT/StringRef.h>

#include "DeviceTraits.hpp"
#include "Logger.hpp"
#include "Utils.hpp"

namespace mneme {
template <typename MemImplT, DeviceVendors VendorTypes> class MnemeMemoryBlob {
public:
  using DeviceError_t = typename DeviceTraits<VendorTypes>::DeviceError_t;
  using DeviceStream_t = typename DeviceTraits<VendorTypes>::DeviceStream_t;
  using KernelFunction_t = typename DeviceTraits<VendorTypes>::KernelFunction_t;
  using MemoryAllocationHandle_t =
      typename DeviceTraits<VendorTypes>::MemoryAllocationHandle_t;

protected:
  uint64_t ActualSize;
  MemoryAllocationHandle_t MemHandle;
  uintptr_t BlobAddr;
  uint64_t Size;
  uintptr_t DeviceID;

public:
  MnemeMemoryBlob() : ActualSize(0), BlobAddr(0), Size(0), DeviceID(0) {}
  hipError_t allocate(uintptr_t Addr, uintptr_t Size, int DeviceID = 0) {
    this->Size = Size;
    this->DeviceID = DeviceID;
    auto MinPageSize = MemImplT::getMinPageSize(DeviceID);
    this->ActualSize = util::roundUp(Size, MinPageSize);
    void *VA = static_cast<MemImplT &>(*this).getVirtualAddress(
        ActualSize, Addr, MinPageSize);
    DBG(Logger::logs("mneme") << "Requested Addr: " << std::hex << Addr
                              << std::dec << " Reserved Addr: " << VA << "\n");

    // We need to pass here "ActualSize". As device allocators depend on page
    // aligned allocations
    static_cast<MemImplT &>(*this).mmap(MemHandle, VA, ActualSize, DeviceID);
    this->BlobAddr = reinterpret_cast<uintptr_t>(VA);
    return hipSuccess;
  };

  hipError_t release() {
    if (!BlobAddr)
      return hipSuccess;
    static_cast<MemImplT &>(*this).unmap(MemHandle, BlobAddr, ActualSize);
    BlobAddr = 0;
    return hipSuccess;
  }

  void *ptr() { return reinterpret_cast<void *>(BlobAddr); }

  MnemeMemoryBlob(const MnemeMemoryBlob &) = delete;
  MnemeMemoryBlob &operator=(const MnemeMemoryBlob &) = delete;

  MnemeMemoryBlob &operator=(MnemeMemoryBlob &&other) noexcept {
    if (this != &other) {
      MemHandle = other.MHandle;
      BlobAddr = other.BlobAddr;
      Size = other.Size;
      ActualSize = other.ActualSize;
      DeviceID = other.DeviceID;
      other.BlobAddr = 0;
    }
    return *this;
  }

  MnemeMemoryBlob(MnemeMemoryBlob &&other) noexcept
      : MemHandle(other.MemHandle), BlobAddr(other.BlobAddr), Size(other.Size),
        ActualSize(other.ActualSize), DeviceID(other.DeviceID) {
    other.BlobAddr = 0;
  }

  template <typename MemImplT_, DeviceVendors VendorTypes_>
  friend llvm::raw_ostream &
  operator<<(llvm::raw_ostream &OS,
             const MnemeMemoryBlob<MemImplT_, VendorTypes_> &Blob);
};

template <typename MemImplT, DeviceVendors VendorTypes>
llvm::raw_ostream &
operator<<(llvm::raw_ostream &OS,
           const MnemeMemoryBlob<MemImplT, VendorTypes> &Blob) {
  // The format in the binary is the following:
  // | Var-Name-Size | Var-Name | Var-Size | Device Address | Var Data |
  OS << llvm::StringRef(reinterpret_cast<const char *>(&Blob.ActualSize),
                        sizeof(Blob.ActualSize));
  OS << llvm::StringRef(reinterpret_cast<const char *>(&Blob.Size),
                        sizeof(Blob.Size));
  OS << llvm::StringRef(reinterpret_cast<const char *>(&Blob.BlobAddr),
                        sizeof(Blob.BlobAddr));
  DBG(Logger::logs("mneme")
      << "Serializing a memory blob of size " << Blob.Size << "\n");
  uint8_t *HostData = new uint8_t[Blob.Size];
  auto EC = DeviceTraits<VendorTypes>::DeviceErrorCheck(MemImplT::DeviceCopy(
      static_cast<void *>(HostData), reinterpret_cast<void *>(Blob.BlobAddr),
      Blob.Size, MemImplT::MemcpyHostToDeviceKind()));
  if (EC)
    FATAL_ERROR("Error in copying data from device when serializing context on "
                "output stream");
  OS << llvm::StringRef(reinterpret_cast<const char *>(HostData), Blob.Size);

  delete[] HostData;
  return OS;
}
} // namespace mneme
