#pragma once

#include <cstdint>
#include <cstring>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <sys/types.h>
#include <utility>

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeUtils.hpp"

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
  size_t ActualSize;
  MemoryAllocationHandle_t MemHandle;
  void *BlobAddr;
  size_t Size;
  int DeviceID;
  std::unique_ptr<uint8_t[]> HostData;
  bool IsMapped;

public:
  MnemeMemoryBlob(size_t ASize = 0, void *BlobAddr = nullptr, size_t SSize = 0)
      : ActualSize(ASize), BlobAddr(BlobAddr), Size(SSize),
        HostData(new uint8_t[SSize]), IsMapped(false), MemHandle{} {
    auto EC =
        MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::getDevice(DeviceID));
    if (EC)
      LOG_FATAL("Could not set device id on memory blob\nEC:{}", EC.value());
  }

  DeviceError_t map(void *VA, size_t ASize, size_t Size) {
    this->Size = Size;
    ActualSize = ASize;
    // We need to pass here "ActualSize". As device allocators depend on page
    // aligned allocations
    MnemeDeviceRT::mmap(MemHandle, VA, ActualSize, DeviceID);
    this->BlobAddr = VA;
    this->IsMapped = true;
    return MnemeDeviceRT::DeviceSuccess;
  };

  DeviceError_t allocate(size_t Size) {
    auto ret = MnemeDeviceRT::DeviceMalloc(&(this->BlobAddr), Size);
    this->ActualSize = Size;
    this->Size = Size;
    this->IsMapped = false;
    return ret;
  }

  DeviceError_t release() {
    if (!BlobAddr)
      return MnemeDeviceRT::DeviceSuccess;

    if (!IsMapped) {
      auto ret = MnemeDeviceRT::DeviceFree(BlobAddr);
      BlobAddr = 0;
      return ret;
    }
    MnemeDeviceRT::unmap(MemHandle, BlobAddr, ActualSize);
    BlobAddr = 0;
    return MnemeDeviceRT::DeviceSuccess;
  }

  void setHostData(std::unique_ptr<uint8_t[]> HostData) {
    this->HostData = std::move(HostData);
  }

  ~MnemeMemoryBlob() {
    if (BlobAddr != 0 && IsMapped) {
      LOG_FATAL("Destroying memory descriptor without releasing device memory");
    }
  }

  static std::pair<void *, MnemeMemoryBlob<VendorTypes>>
  fromBuffer(const char *&Buffer) {
    size_t ActualSize = util::extractScalar<size_t>(Buffer);
    size_t Size = util::extractScalar<size_t>(Buffer);
    void *DeviceAddr = util::extractScalar<void *>(Buffer);
    auto Blob = MnemeMemoryBlob<VendorTypes>(ActualSize, 0, Size);
    std::memcpy(Blob.getHostData().get(), Buffer, Size);
    Buffer += Size;
    LOG_DEBUG("Read memory blob at address {} SIZE: {} ActualSize:{}",
              DeviceAddr, Size, ActualSize);
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
  size_t getActualSize() const { return ActualSize; }
  size_t getSize() const { return Size; }
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

  LOG_DEBUG("Serializing MemoryBlob, DevAddr:{} MirroredHostAddr:{} Size:{} "
            "ActualSize:{}",
            (void *)Blob.BlobAddr,
            reinterpret_cast<void *>(Blob.getHostData().get()), Blob.getSize(),
            Blob.getActualSize());
  auto EC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
      DeviceTraits<VendorTypes>::DeviceCopy(
          static_cast<void *>(Blob.getHostData().get()),
          reinterpret_cast<void *>(Blob.BlobAddr), Blob.Size,
          DeviceTraits<VendorTypes>::MemcpyHostToDeviceKind()));
  if (EC)
    LOG_FATAL("Error in copying data from device when serializing context on "
              "output stream\nDevice Error Msg: " +
              EC.value() + "\n");
  OS << llvm::StringRef(
      reinterpret_cast<const char *>(Blob.getHostData().get()), Blob.Size);

  return OS;
}
} // namespace mneme
