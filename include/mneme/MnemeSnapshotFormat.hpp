#pragma once
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeKernelInfo.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeMemory.hpp"
#include "mneme/MnemeSnapshotRecords.hpp"
#include "mneme/MnemeUtils.hpp"

namespace mneme {

struct ReplayGlobalVar {
  void *HostAddr;
  void *DevAddr;
  uint64_t VarSize;
  ReplayGlobalVar(void *DevAddr, uint64_t VarSize)
      : HostAddr(new uint8_t[VarSize]), DevAddr(DevAddr), VarSize(VarSize) {}
  ReplayGlobalVar(void *HostAddr, void *DevAddr, uint64_t VarSize)
      : HostAddr(HostAddr), DevAddr(DevAddr), VarSize(VarSize) {}
  ReplayGlobalVar() = delete;
  ~ReplayGlobalVar() {
    if (HostAddr)
      delete[] static_cast<uint8_t *>(HostAddr);
  }

  ReplayGlobalVar(const ReplayGlobalVar &) = delete;
  ReplayGlobalVar &operator=(const ReplayGlobalVar &) = delete;

  ReplayGlobalVar(ReplayGlobalVar &&Other)
      : HostAddr(Other.HostAddr), DevAddr(Other.DevAddr),
        VarSize(Other.VarSize) {
    Other.HostAddr = nullptr;
  }

  ReplayGlobalVar &operator=(ReplayGlobalVar &&Other) {
    if (this != &Other) {
      this->HostAddr = Other.HostAddr;
      this->DevAddr = Other.DevAddr;
      this->VarSize = Other.VarSize;
      Other.HostAddr = nullptr;
    }
    return *this;
  }
};

// The in-memory contents of a snapshot, as produced by the snapshot readers
// and consumed by the replay memory state constructors.
template <DeviceVendors VendorTypes> struct Snapshot {
  std::shared_ptr<KernelInfo> KInfo;
  std::unordered_map<std::string, ReplayGlobalVar> GlobalVars;
  llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> DeviceMemory;
};

inline std::pair<std::string, ReplayGlobalVar>
readGlobalVarRecord(const char *&Buffer) {
  GlobalVarHeader Header = GlobalVarHeader::read(Buffer);
  ReplayGlobalVar RGV(Header.DevAddr, Header.Size);
  std::memcpy(const_cast<void *>(RGV.HostAddr), Buffer, Header.Size);
  Buffer += Header.Size;
  LOG_DEBUG("Loaded from buffer Global, Name:{}, VarSize:{}, RecoredAddr:{}",
            Header.Name, Header.Size, Header.DevAddr);
  return std::pair<std::string, ReplayGlobalVar>(std::move(Header.Name),
                                                 std::move(RGV));
}

// A host copy of every global a prologue captured, keyed by variable name. The
// diff writer uses it as the base its ranges are computed against.
using GlobalSnapshotData =
    std::unordered_map<std::string, std::vector<uint8_t>>;

enum class SnapshotKind : uint32_t { Bytes = 1, Diff = 2 };

// The only code in Mneme that knows a snapshot magic string.
struct SnapshotHeader {
  SnapshotKind Kind;
  uint32_t Version;

  static constexpr char Magic[8] = {'M', 'N', 'E', 'M', 'E', 'S', 'N', 'P'};
  static constexpr size_t Size = 16;

  // Returns the header and the number of prefix bytes the payload follows.
  static std::pair<SnapshotHeader, size_t> parse(llvm::StringRef Buffer);
  void write(llvm::raw_ostream &OS) const;

private:
  // Diff files written before the container header carried this instead.
  static constexpr char LegacyDiffMagic[] = "MNEME_DIFF_V1";
  static constexpr size_t LegacyDiffMagicSize = sizeof(LegacyDiffMagic) - 1;
};

inline std::pair<SnapshotHeader, size_t>
SnapshotHeader::parse(llvm::StringRef Buffer) {
  if (Buffer.size() >= Size && Buffer.take_front(sizeof(Magic)) ==
                                   llvm::StringRef(Magic, sizeof(Magic))) {
    SnapshotKind Kind;
    uint32_t Version;
    std::memcpy(&Kind, Buffer.data() + sizeof(Magic), sizeof(Kind));
    std::memcpy(&Version, Buffer.data() + sizeof(Magic) + sizeof(Kind),
                sizeof(Version));
    return {SnapshotHeader{Kind, Version}, Size};
  }

  if (Buffer.size() >= LegacyDiffMagicSize &&
      Buffer.take_front(LegacyDiffMagicSize) ==
          llvm::StringRef(LegacyDiffMagic, LegacyDiffMagicSize))
    return {SnapshotHeader{SnapshotKind::Diff, 1}, LegacyDiffMagicSize};

  return {SnapshotHeader{SnapshotKind::Bytes, 0}, 0};
}

// Still emits the legacy prefixes so files written here stay readable by older
// Mneme builds; the container header replaces this in a later change.
inline void SnapshotHeader::write(llvm::raw_ostream &OS) const {
  if (Kind == SnapshotKind::Diff)
    util::writeBytes(OS, llvm::StringRef(LegacyDiffMagic, LegacyDiffMagicSize));
}

template <DeviceVendors VendorTypes> class BaseSnapshotSource;

// An opened snapshot file. Each subclass decodes one on-disk layout.
template <DeviceVendors VendorTypes> class SnapshotReader {
public:
  SnapshotReader(std::string Filename,
                 std::unique_ptr<llvm::MemoryBuffer> Buffer,
                 size_t PayloadOffset)
      : Filename(std::move(Filename)), Buffer(std::move(Buffer)),
        PayloadOffset(PayloadOffset) {}
  virtual ~SnapshotReader() = default;

  // True if read() never consults Base.
  virtual bool isSelfContained() const = 0;

  virtual Snapshot<VendorTypes>
  read(const std::string &KernelName,
       const BaseSnapshotSource<VendorTypes> &Base) const = 0;

protected:
  const char *payload() const {
    return Buffer->getBufferStart() + PayloadOffset;
  }

  std::string Filename;
  std::unique_ptr<llvm::MemoryBuffer> Buffer;
  size_t PayloadOffset;
};

// Loads a base prologue snapshot on demand for readers that need one.
template <DeviceVendors VendorTypes> class BaseSnapshotSource {
public:
  // An empty Filename means no base is available.
  explicit BaseSnapshotSource(std::string Filename = "")
      : Filename(std::move(Filename)) {}

  bool empty() const { return Filename.empty(); }

  Snapshot<VendorTypes> load(const std::string &KernelName) const;

private:
  std::string Filename;
};

template <DeviceVendors VendorTypes>
class BytesReaderV0 : public SnapshotReader<VendorTypes> {
public:
  using SnapshotReader<VendorTypes>::SnapshotReader;

  bool isSelfContained() const override { return true; }

  Snapshot<VendorTypes>
  read(const std::string &KernelName,
       const BaseSnapshotSource<VendorTypes> &) const override {
    Snapshot<VendorTypes> Snap;
    // KernelInfo's constructor takes a non-const std::string &, so name it with
    // a mutable local.
    std::string Name = KernelName;
    Snap.KInfo = std::make_shared<KernelInfo>(Name);

    auto &GlobalVars = Snap.GlobalVars;
    auto &DeviceMemory = Snap.DeviceMemory;
    auto &KInfo = Snap.KInfo;

    auto *Start = this->payload();
    auto *CurrentPtr = Start;
    size_t TotalGlobals = util::extractScalar<size_t>(CurrentPtr);
    LOG_DEBUG("Snapshot contains {} Globals at location {}", TotalGlobals,
              (uintptr_t)CurrentPtr - (uintptr_t)Start);
    for (auto I = 0; I < TotalGlobals; I++) {
      auto [Name, RGV] = readGlobalVarRecord(CurrentPtr);
      GlobalVars.try_emplace(Name, std::move(RGV));
    }

    auto TotalMemBlobs = util::extractScalar<size_t>(CurrentPtr);

    LOG_DEBUG("Snapshot contains {} Memory Blobs starting at location {}",
              TotalMemBlobs, (uintptr_t)CurrentPtr - (uintptr_t)Start);

    for (auto M = 0; M < TotalMemBlobs; M++) {
      DeviceMemory.insert(MnemeMemoryBlob<VendorTypes>::fromBuffer(CurrentPtr));
    }

    // Get kernel arguments.
    auto TotalArguments = util::extractScalar<size_t>(CurrentPtr);
    LOG_DEBUG("Snapshot contains {} total arguments starting at location {}",
              TotalArguments, (uintptr_t)CurrentPtr - (uintptr_t)Start);
    KInfo->KernelArgSizes.resize(TotalArguments);
    KInfo->ArgData.resize(TotalArguments);
    for (auto A = 0; A < TotalArguments; A++) {
      KInfo->KernelArgSizes[A] = util::extractScalar<size_t>(CurrentPtr);
      KInfo->setArgData(CurrentPtr, A);
    }

    return Snap;
  }
};

template <DeviceVendors VendorTypes>
class DiffReaderV1 : public SnapshotReader<VendorTypes> {
public:
  using SnapshotReader<VendorTypes>::SnapshotReader;

  bool isSelfContained() const override { return false; }

  Snapshot<VendorTypes>
  read(const std::string &KernelName,
       const BaseSnapshotSource<VendorTypes> &Base) const override {
    if (Base.empty())
      LOG_FATAL("Mneme diff snapshot " + this->Filename +
                " requires a base prologue snapshot");

    // A diff stores only changed ranges, so reconstruct the full base prologue
    // first and then overlay the diff onto it.
    Snapshot<VendorTypes> Snap = Base.load(KernelName);
    const std::string &Filename = this->Filename;
    auto &GlobalVars = Snap.GlobalVars;
    auto &DeviceMemory = Snap.DeviceMemory;

    auto *CurrentPtr = this->payload();
    size_t TotalGlobals = util::extractScalar<size_t>(CurrentPtr);
    if (TotalGlobals != GlobalVars.size())
      LOG_FATAL("Mneme diff " + Filename +
                " does not match prologue global count");

    for (size_t I = 0; I < TotalGlobals; ++I) {
      GlobalVarHeader GVH = GlobalVarHeader::read(CurrentPtr);
      size_t NumRanges = util::extractScalar<size_t>(CurrentPtr);

      auto It = GlobalVars.find(GVH.Name);
      if (It == GlobalVars.end())
        LOG_FATAL("Mneme diff references global missing from prologue: " +
                  GVH.Name);
      if (It->second.VarSize != GVH.Size)
        LOG_FATAL("Mneme diff global size mismatch for: " + GVH.Name);
      It->second.DevAddr = GVH.DevAddr;
      applyDiffRanges(
          CurrentPtr,
          llvm::MutableArrayRef<uint8_t>(
              static_cast<uint8_t *>(It->second.HostAddr), It->second.VarSize),
          NumRanges);
    }

    size_t TotalMemBlobs = util::extractScalar<size_t>(CurrentPtr);
    if (TotalMemBlobs != DeviceMemory.size())
      LOG_FATAL("Mneme diff " + Filename +
                " does not match prologue memory blob count");

    for (size_t I = 0; I < TotalMemBlobs; ++I) {
      BlobHeader BH = BlobHeader::read(CurrentPtr);
      auto MD = metadata::fromBuffer(CurrentPtr);
      size_t NumRanges = util::extractScalar<size_t>(CurrentPtr);

      auto It = DeviceMemory.find(BH.DevAddr);
      if (It == DeviceMemory.end())
        LOG_FATAL("Mneme diff references device allocation missing from "
                  "prologue");
      auto &Blob = It->second;
      if (Blob.getActualSize() != BH.ActualSize || Blob.getSize() != BH.Size)
        LOG_FATAL("Mneme diff memory blob size mismatch");
      Blob.setMetadata(MD);
      applyDiffRanges(CurrentPtr,
                      llvm::MutableArrayRef<uint8_t>(Blob.getHostData().get(),
                                                     Blob.getSize()),
                      NumRanges);
    }

    return Snap;
  }

private:
  static void applyDiffRanges(const char *&Buffer,
                              llvm::MutableArrayRef<uint8_t> Target,
                              size_t NumRanges) {
    for (size_t R = 0; R < NumRanges; ++R) {
      size_t Offset = util::extractScalar<size_t>(Buffer);
      size_t Size = util::extractScalar<size_t>(Buffer);
      if (Offset > Target.size() || Size > Target.size() - Offset)
        LOG_FATAL("Malformed Mneme diff range: offset " +
                  std::to_string(Offset) + " size " + std::to_string(Size) +
                  " exceeds target size " + std::to_string(Target.size()));
      std::memcpy(Target.data() + Offset, Buffer, Size);
      Buffer += Size;
    }
  }
};

// Every layout Mneme has ever written needs a row here.
template <DeviceVendors VendorTypes> class SnapshotFormatRegistry {
public:
  static std::unique_ptr<SnapshotReader<VendorTypes>>
  open(const std::string &Filename) {
    LOG_DEBUG("Opening snapshot file {}", Filename);
    auto Buffer = openSnapshotFile(Filename);
    auto [Header, PayloadOffset] = SnapshotHeader::parse(Buffer->getBuffer());

    size_t Count = 0;
    const Entry *Table = table(Count);
    for (size_t I = 0; I < Count; ++I)
      if (Table[I].Kind == Header.Kind && Table[I].Version == Header.Version)
        return Table[I].Make(Filename, std::move(Buffer), PayloadOffset);

    LOG_FATAL("Unsupported Mneme snapshot format in " + Filename + ": kind " +
              std::to_string(static_cast<uint32_t>(Header.Kind)) + " version " +
              std::to_string(Header.Version));
  }

private:
  using Factory = std::unique_ptr<SnapshotReader<VendorTypes>> (*)(
      std::string, std::unique_ptr<llvm::MemoryBuffer>, size_t);

  struct Entry {
    SnapshotKind Kind;
    uint32_t Version;
    Factory Make;
  };

  template <typename ReaderT>
  static std::unique_ptr<SnapshotReader<VendorTypes>>
  make(std::string Filename, std::unique_ptr<llvm::MemoryBuffer> Buffer,
       size_t PayloadOffset) {
    return std::make_unique<ReaderT>(std::move(Filename), std::move(Buffer),
                                     PayloadOffset);
  }

  static const Entry *table(size_t &Count) {
    static const Entry Table[] = {
        {SnapshotKind::Bytes, 0, &make<BytesReaderV0<VendorTypes>>},
        {SnapshotKind::Diff, 1, &make<DiffReaderV1<VendorTypes>>},
    };
    Count = sizeof(Table) / sizeof(Table[0]);
    return Table;
  }

  static std::unique_ptr<llvm::MemoryBuffer>
  openSnapshotFile(const std::string &Filename) {
    if (!std::filesystem::exists(Filename))
      LOG_FATAL("Mneme Snapshot file does not exist");

    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> BufferOrErr =
        llvm::MemoryBuffer::getFile(Filename);
    if (std::error_code EC = BufferOrErr.getError())
      LOG_FATAL("Error when opening file " + EC.message());

    return std::move(BufferOrErr.get());
  }
};

template <DeviceVendors VendorTypes>
Snapshot<VendorTypes>
BaseSnapshotSource<VendorTypes>::load(const std::string &KernelName) const {
  if (empty())
    LOG_FATAL("No base snapshot was provided");

  auto Reader = SnapshotFormatRegistry<VendorTypes>::open(Filename);
  if (!Reader->isSelfContained())
    LOG_FATAL("Snapshot " + Filename +
              " cannot be used as a base because it is not self-contained");

  return Reader->read(KernelName, BaseSnapshotSource<VendorTypes>());
}

} // namespace mneme
