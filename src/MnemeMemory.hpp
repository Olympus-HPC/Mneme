#pragma once

#include <cstdint>
#include <cstring>
#include <hip/hip_runtime.h>
#include <iostream>
#include <llvm/ADT/StringRef.h>
#include <memory>
#include <sys/types.h>
#include <utility>

#include "DeviceTraits.hpp"
#include "Logger.hpp"
#include "Utils.hpp"

namespace mneme {
template <DeviceVendors VendorTypes> class MnemeMemoryBlob {
public:
  using MnemeDeviceRT = DeviceTraits<VendorTypes>;
  using DeviceError_t = typename MnemeDeviceRT::DeviceError_t;
  using DeviceStream_t = typename MnemeDeviceRT::DeviceStream_t;
  using KernelFunction_t = typename MnemeDeviceRT::KernelFunction_t;
  using MemoryAllocationHandle_t =
      typename MnemeDeviceRT::MemoryAllocationHandle_t;

protected:
  uint64_t ActualSize;
  MemoryAllocationHandle_t MemHandle;
  void *BlobAddr;
  uint64_t Size;
  uint64_t DeviceID;
  std::unique_ptr<uint8_t[]> HostData;
  bool IsMapped;

public:
  MnemeMemoryBlob(uint64_t ActualSize = 0, void *BlobAddr = nullptr,
                  uint64_t Size = 0, uint64_t DeviceID = 0)
      : ActualSize(ActualSize), BlobAddr(BlobAddr), Size(Size), DeviceID(0),
        HostData(nullptr), IsMapped(false) {}

  hipError_t allocate(void *Addr, uintptr_t Size, int DeviceID = 0) {
    this->Size = Size;
    this->DeviceID = DeviceID;
    auto MinPageSize = MnemeDeviceRT::getMinPageSize(DeviceID);
    this->ActualSize = util::roundUp(Size, MinPageSize);
    void *VA = MnemeDeviceRT::getVirtualAddress(ActualSize, Addr, MinPageSize);
    DBG(Logger::logs("mneme") << "Requested Addr: " << std::hex << Addr
                              << std::dec << " Reserved Addr: " << VA << "\n");

    // We need to pass here "ActualSize". As device allocators depend on page
    // aligned allocations
    MnemeDeviceRT::mmap(MemHandle, VA, ActualSize, DeviceID);
    this->BlobAddr = VA;
    this->IsMapped = true;

    HostData = std::unique_ptr<uint8_t[]>(new uint8_t[Size]);
    std::cout << "Allocated Device Address at "
              << reinterpret_cast<void *>(HostData.get()) << "\n";

    return hipSuccess;
  };

  hipError_t release() {
    if (!BlobAddr)
      return hipSuccess;
    if (IsMapped)
      MnemeDeviceRT::unmap(MemHandle, BlobAddr, ActualSize);
    BlobAddr = 0;
    return hipSuccess;
  }

  void setHostData(std::unique_ptr<uint8_t[]> HostData) {
    this->HostData = std::move(HostData);
  }

  ~MnemeMemoryBlob() {
    if (BlobAddr != 0 && IsMapped) {
      FATAL_ERROR(
          "Destroying memory descriptor without releasing device memory");
    }
  }

  static std::pair<void *, MnemeMemoryBlob<VendorTypes>>
  fromBuffer(const char *&Buffer) {
    size_t ActualSize = util::extractScalar<size_t>(Buffer);
    size_t Size = util::extractScalar<size_t>(Buffer);
    void *DeviceAddr = util::extractScalar<void *>(Buffer);
    std::unique_ptr<uint8_t[]> HostData = std::make_unique<uint8_t[]>(Size);
    std::memcpy(HostData.get(), Buffer, Size);
    Buffer += Size;
    auto Blob = MnemeMemoryBlob<VendorTypes>(ActualSize, 0, Size, 0);
    Blob.setHostData(std::move(HostData));
    DBG(Logger::logs("mneme")
        << "Read memory blob at address " << std::hex << DeviceAddr << std::dec
        << " of size " << Size << " and ActualSize is " << ActualSize << "\n");
    return std::make_pair(DeviceAddr, std::move(Blob));
  }

  void *ptr() { return reinterpret_cast<void *>(BlobAddr); }

  MnemeMemoryBlob(const MnemeMemoryBlob &) = delete;
  MnemeMemoryBlob &operator=(const MnemeMemoryBlob &) = delete;

  MnemeMemoryBlob &operator=(MnemeMemoryBlob &&other) noexcept {
    if (this != &other) {
      MemHandle = other.MemHandle;
      BlobAddr = other.BlobAddr;
      Size = other.Size;
      ActualSize = other.ActualSize;
      DeviceID = other.DeviceID;
      HostData = std::move(other.HostData);
      IsMapped = other.IsMapped;
      other.BlobAddr = 0;
      other.HostData = nullptr;
    }
    return *this;
  }

  MnemeMemoryBlob(MnemeMemoryBlob &&other) noexcept
      : MemHandle(other.MemHandle), BlobAddr(other.BlobAddr), Size(other.Size),
        ActualSize(other.ActualSize), DeviceID(other.DeviceID),
        HostData(std::move(other.HostData)), IsMapped(other.IsMapped) {
    other.BlobAddr = 0;
    other.HostData = nullptr;
  }

  template <DeviceVendors VendorTypes_>
  friend llvm::raw_ostream &
  operator<<(llvm::raw_ostream &OS, const MnemeMemoryBlob<VendorTypes_> &Blob);

  void *getBlobAddr() const { return BlobAddr; }
  uint64_t getActualSize() const { return ActualSize; }
  uint64_t getSize() const { return Size; }
  const std::unique_ptr<uint8_t[]> &getHostData() const { return HostData; }
};

template <DeviceVendors VendorTypes>
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const MnemeMemoryBlob<VendorTypes> &Blob) {
  // The format in the binary is the following:
  // | Var-Name-Size | Var-Name | Var-Size | Device Address | Var Data |
  OS << llvm::StringRef(reinterpret_cast<const char *>(&Blob.ActualSize),
                        sizeof(Blob.ActualSize));
  OS << llvm::StringRef(reinterpret_cast<const char *>(&Blob.Size),
                        sizeof(Blob.Size));
  OS << llvm::StringRef(reinterpret_cast<const char *>(&Blob.BlobAddr),
                        sizeof(Blob.BlobAddr));

  DBG(Logger::logs("mneme")
      << "Serializing a memory blob of size " << Blob.Size
      << " of  device address " << std::hex << (void *)Blob.BlobAddr << std::dec
      << " and host address '" << std::hex
      << reinterpret_cast<void *>(Blob.getHostData().get()) << std::dec << "'"
      << std::endl);
  auto EC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
      DeviceTraits<VendorTypes>::DeviceCopy(
          static_cast<void *>(Blob.getHostData().get()),
          reinterpret_cast<void *>(Blob.BlobAddr), Blob.Size,
          DeviceTraits<VendorTypes>::MemcpyHostToDeviceKind()));
  if (EC)
    FATAL_ERROR("Error in copying data from device when serializing context on "
                "output stream\nDevice Error Msg: " +
                EC.value() + "\n");
  OS << llvm::StringRef(
      reinterpret_cast<const char *>(Blob.getHostData().get()), Blob.Size);

  return OS;
}
} // namespace mneme
