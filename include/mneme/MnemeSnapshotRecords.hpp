#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

#include "mneme/MnemeLLVMUtils.hpp"
#include "mneme/MnemeUtils.hpp"

namespace mneme {

// Global-variable record prefix: | Name length | Name | Size | DevAddr |.
struct GlobalVarHeader {
  std::string Name;
  size_t Size;
  void *DevAddr;

  void write(llvm::raw_ostream &OS) const {
    util::writeScalar(OS, Name.size());
    util::writeBytes(OS, llvm::StringRef(Name));
    util::writeScalar(OS, Size);
    util::writeScalar(OS, DevAddr);
  }

  static GlobalVarHeader read(const char *&Buffer) {
    std::string Name = util::readSizedString(Buffer);
    size_t Size = util::extractScalar<size_t>(Buffer);
    void *DevAddr = util::extractScalar<void *>(Buffer);
    return GlobalVarHeader{std::move(Name), Size, DevAddr};
  }

  size_t serializedSize() const {
    return sizeof(size_t) + Name.size() + sizeof(Size) + sizeof(DevAddr);
  }
};

// Device-memory blob record prefix: | ActualSize | Size | BlobId | BlobOffset
// |. BlobOffset is relative to the recorded VA reservation so that replay can
// place the blob at any reservation base. Metadata is excluded because the
// bytes and diff layouts place it differently.
struct BlobHeader {
  size_t ActualSize;
  size_t Size;
  uint64_t BlobId;
  uint64_t BlobOffset;

  void write(llvm::raw_ostream &OS) const {
    util::writeScalar(OS, ActualSize);
    util::writeScalar(OS, Size);
    util::writeScalar(OS, BlobId);
    util::writeScalar(OS, BlobOffset);
  }

  static BlobHeader read(const char *&Buffer) {
    size_t ActualSize = util::extractScalar<size_t>(Buffer);
    size_t Size = util::extractScalar<size_t>(Buffer);
    uint64_t BlobId = util::extractScalar<uint64_t>(Buffer);
    uint64_t BlobOffset = util::extractScalar<uint64_t>(Buffer);
    return BlobHeader{ActualSize, Size, BlobId, BlobOffset};
  }

  static constexpr size_t serializedSize() {
    return 2 * sizeof(size_t) + 2 * sizeof(uint64_t);
  }
};

} // namespace mneme
