#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <optional>
#include <regex>
#include <unordered_map>
#include <vector>

#include "llvm/Demangle/Demangle.h"
#include <llvm/ADT/StableHashing.h>
#include <llvm/ADT/StringRef.h>
#include <string>
#include <sys/types.h>

#include "proteus/impl/CompilerInterfaceDevice.h"
#include "proteus/impl/Hashing.h"
#include <proteus/impl/JitEngineDevice.h>

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeKernelInfo.hpp"
#include "mneme/MnemeLLVMUtils.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeMemory.hpp"
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

template <DeviceVendors VendorTypes> class MnemeSnapshot {
  using MnemeDeviceRT = DeviceTraits<VendorTypes>;
  using DeviceError_t = typename MnemeDeviceRT::DeviceError_t;
  using DeviceStream_t = typename MnemeDeviceRT::DeviceStream_t;
  using KernelFunction_t = typename MnemeDeviceRT::KernelFunction_t;
  static constexpr const char DiffMagic[] = "MNEME_DIFF_V1";
  static constexpr size_t DiffMagicSize = sizeof(DiffMagic) - 1;
  static constexpr size_t DiffChunkSize = 1 << 20;


  static bool isDiffBuffer(llvm::StringRef Buffer) {
    return Buffer.size() >= DiffMagicSize &&
           Buffer.take_front(DiffMagicSize) == llvm::StringRef(DiffMagic);
  }

  static size_t countChangedRanges(const uint8_t *Base, const uint8_t *Current,
                                   size_t Size) {
    // Count the number of contiguous ranges that have changed between Base and
    // Current. We want to write out the number of ranges so that the reader 
    // can know how many ranges to read.
    size_t Count = 0;
    bool InRange = false;
    for (size_t I = 0; I < Size; ++I) {
      if (Base[I] != Current[I]) {
        if (!InRange) {
          Count++;
          InRange = true;
        }
      } else {
        InRange = false;
      }
    }
    return Count;
  }

  static void writeChangedRanges(llvm::raw_ostream &OS, const uint8_t *Base,
                                 const uint8_t *Current, size_t Size,
                                 size_t BaseOffset,
                                 uint8_t *UpdateBase = nullptr) {
    // Write out the contiguous ranges that have changed between Base 
    // and Current.
    size_t I = 0;
    while (I < Size) {
      while (I < Size && Base[I] == Current[I])
        ++I;
      if (I == Size)
        break;

      size_t Start = I;
      while (I < Size && Base[I] != Current[I])
        ++I;

      size_t Offset = BaseOffset + Start;
      size_t Len = I - Start;
      util::writeScalar(OS, Offset);
      util::writeScalar(OS, Len);
      util::writeBytes(OS, Current + Start, Len);
      if (UpdateBase)
        std::memcpy(UpdateBase + Start, Current + Start, Len);
    }
  }

  static void writeCountAndWriteChangedRanges(
      llvm::raw_ostream &OS, MnemeMemoryBlob<VendorTypes> &Blob) {
    auto Size = Blob.getSize();

    // early exit
    if (Size == 0) {
      size_t NumRanges = 0;
      util::writeScalar(OS, NumRanges);
      return;
    }

    std::unique_ptr<uint8_t[]> Scratch(new uint8_t[DiffChunkSize]);
    llvm::SmallVector<char, 0> DiffBytes;
    llvm::raw_svector_ostream DiffOS(DiffBytes);
    auto *Base = Blob.getHostData().get();
    auto *DevAddr = static_cast<uint8_t *>(Blob.getBlobAddr());
    size_t NumRanges = 0;

    for (size_t Offset = 0; Offset < Size; Offset += DiffChunkSize) {
      size_t ChunkSize = std::min(DiffChunkSize, Size - Offset);
      auto EC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
          DeviceTraits<VendorTypes>::DeviceCopy(
              Scratch.get(), DevAddr + Offset, ChunkSize,
              DeviceTraits<VendorTypes>::MemcpyDeviceToHostKind()));
      if (EC)
        LOG_FATAL("Error in copying data from device when writing diff\n"
                  "Device Error Msg: " +
                  EC.value() + "\n");

      size_t I = 0;
      auto *ChunkBase = Base + Offset;
      while (I < ChunkSize) {
        while (I < ChunkSize && ChunkBase[I] == Scratch[I])
          ++I;
        if (I == ChunkSize)
          break;

        size_t Start = I;
        while (I < ChunkSize && ChunkBase[I] != Scratch[I])
          ++I;

        size_t RangeOffset = Offset + Start;
        size_t Len = I - Start;
        util::writeScalar(DiffOS, RangeOffset);
        util::writeScalar(DiffOS, Len);
        util::writeBytes(DiffOS, Scratch.get() + Start, Len);
        std::memcpy(ChunkBase + Start, Scratch.get() + Start, Len);
        ++NumRanges;
      }
    }

    util::writeScalar(OS, NumRanges);
    util::writeBytes(OS, DiffBytes.data(), DiffBytes.size());
  }

  static void applyDiffRanges(const char *&Buffer, uint8_t *Target,
                              size_t TargetSize, size_t NumRanges) {
    for (size_t R = 0; R < NumRanges; ++R) {
      size_t Offset = util::extractScalar<size_t>(Buffer);
      size_t Size = util::extractScalar<size_t>(Buffer);
      if (Offset > TargetSize || Size > TargetSize - Offset)
        LOG_FATAL("Malformed Mneme diff range: offset " +
                  std::to_string(Offset) + " size " + std::to_string(Size) +
                  " exceeds target size " + std::to_string(TargetSize));
      std::memcpy(Target + Offset, Buffer, Size);
      Buffer += Size;
    }
  }

  static void readFullMnemeSnapShot(
      llvm::MemoryBuffer *Buffer,
      std::unordered_map<std::string, ReplayGlobalVar> &GlobalVars,
      llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &DeviceMemory,
      std::shared_ptr<KernelInfo> KInfo) {
    auto *Start = Buffer->getBufferStart();
    auto *CurrentPtr = Start;
    size_t TotalGlobals = util::extractScalar<size_t>(CurrentPtr);
    LOG_DEBUG("Snapshot contains {} Globals at location {}", TotalGlobals,
              (uintptr_t)CurrentPtr - (uintptr_t)Start);
    for (auto I = 0; I < TotalGlobals; I++) {
      auto [Name, RGV] = fromBuffer(CurrentPtr);
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
  }

  static void readDiffMnemeSnapShot(
      std::string Filename, std::string BaseSnapshotName,
      std::unordered_map<std::string, ReplayGlobalVar> &GlobalVars,
      llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &DeviceMemory,
      std::shared_ptr<KernelInfo> KInfo, llvm::MemoryBuffer *DiffBuffer) {
    readMnemeSnapShot(BaseSnapshotName, GlobalVars, DeviceMemory, KInfo);

    auto *Start = DiffBuffer->getBufferStart();
    auto *CurrentPtr = Start + DiffMagicSize;
    size_t TotalGlobals = util::extractScalar<size_t>(CurrentPtr);
    if (TotalGlobals != GlobalVars.size())
      LOG_FATAL("Mneme diff " + Filename +
                " does not match prologue global count");

    for (size_t I = 0; I < TotalGlobals; ++I) {
      std::string Name = util::readSizedString(CurrentPtr);
      size_t VarSize = util::extractScalar<size_t>(CurrentPtr);
      void *DevAddr = util::extractScalar<void *>(CurrentPtr);
      size_t NumRanges = util::extractScalar<size_t>(CurrentPtr);

      auto It = GlobalVars.find(Name);
      if (It == GlobalVars.end())
        LOG_FATAL("Mneme diff references global missing from prologue: " +
                  Name);
      if (It->second.VarSize != VarSize)
        LOG_FATAL("Mneme diff global size mismatch for: " + Name);
      It->second.DevAddr = DevAddr;
      applyDiffRanges(CurrentPtr, static_cast<uint8_t *>(It->second.HostAddr),
                      It->second.VarSize, NumRanges);
    }

    size_t TotalMemBlobs = util::extractScalar<size_t>(CurrentPtr);
    if (TotalMemBlobs != DeviceMemory.size())
      LOG_FATAL("Mneme diff " + Filename +
                " does not match prologue memory blob count");

    for (size_t I = 0; I < TotalMemBlobs; ++I) {
      size_t ActualSize = util::extractScalar<size_t>(CurrentPtr);
      size_t Size = util::extractScalar<size_t>(CurrentPtr);
      void *DeviceAddr = util::extractScalar<void *>(CurrentPtr);
      auto MD = metadata::fromBuffer(CurrentPtr);
      size_t NumRanges = util::extractScalar<size_t>(CurrentPtr);

      auto It = DeviceMemory.find(DeviceAddr);
      if (It == DeviceMemory.end())
        LOG_FATAL("Mneme diff references device allocation missing from "
                  "prologue");
      auto &Blob = It->second;
      if (Blob.getActualSize() != ActualSize || Blob.getSize() != Size)
        LOG_FATAL("Mneme diff memory blob size mismatch");
      Blob.setMetadata(MD);
      applyDiffRanges(CurrentPtr, Blob.getHostData().get(), Blob.getSize(),
                      NumRanges);
    }
  }

public:
  using GlobalSnapshotData = std::unordered_map<std::string, std::vector<uint8_t>>;

  enum class SnapshotType: uint8_t { Bytes = 0, Diff = 1 };

  static std::pair<std::string, ReplayGlobalVar>
  fromBuffer(const char *&Buffer) {
    std::string Name = util::readSizedString(Buffer);
    size_t VarSize = util::extractScalar<size_t>(Buffer);
    void *DevAddr = util::extractScalar<void *>(Buffer);
    ReplayGlobalVar RGV(DevAddr, VarSize);
    std::memcpy(const_cast<void *>(RGV.HostAddr), Buffer, VarSize);
    Buffer += VarSize;
    LOG_DEBUG("Loaded from buffer Global, Name:{}, VarSize:{}, RecoredAddr:{}",
              Name, VarSize, DevAddr);
    return std::pair<std::string, ReplayGlobalVar>(std::move(Name),
                                                   std::move(RGV));
  }

  std::filesystem::path static takeMnemeBytesSnapshot(
      std::unordered_map<std::string, proteus::GlobalVarInfo> &GlobalVars,
      llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &DeviceMemory,
      std::filesystem::path &Filename,
      llvm::SmallVector<size_t> &KernelArgSizes, void **Args,
      DeviceStream_t Stream, GlobalSnapshotData *CapturedGlobals = nullptr) {
    LOG_DEBUG("Storing mneme snapshot: {}", Filename.string());
    std::error_code EC;
    // Syncrhonize cause we need to get a consistent GPU state.
    // We may want to do a DeviceSynchronize().
    auto DEC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
        DeviceTraits<VendorTypes>::DeviceStreamSynchronize(Stream));
    if (DEC)
      LOG_FATAL("Synnchronizing stream  failed");
    llvm::raw_fd_ostream OutBC(Filename.string(), EC);
    // First write Global Variables.
    size_t TotalGlobals = GlobalVars.size();
    OutBC << llvm::StringRef(reinterpret_cast<const char *>(&TotalGlobals),
                             sizeof(size_t));

    LOG_DEBUG("Number of Globals in snapshot:{} stored at position:{}",
              TotalGlobals, OutBC.tell());

    for (const auto &[VarName, GV] : GlobalVars) {
      std::cout << "Reading " << VarName << " " << GV.HostAddr << " "
                << GV.DevAddr << " " << GV.VarSize << "\n";
      std::unique_ptr<uint8_t[]> HostData(new uint8_t[GV.VarSize]);
      auto DEC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
          DeviceTraits<VendorTypes>::DeviceCopy(
              HostData.get(), const_cast<void *>(GV.DevAddr), GV.VarSize,
              DeviceTraits<VendorTypes>::MemcpyDeviceToHostKind()));
      if (DEC) {
        std::cout << DEC.value() << "\n";
        LOG_FATAL("Copying from device to host for global variables failed\n");
      }

      size_t StrLen = VarName.size();
      OutBC << llvm::StringRef(reinterpret_cast<const char *>(&StrLen),
                               sizeof(StrLen));
      OutBC << VarName;
      OutBC << llvm::StringRef(reinterpret_cast<const char *>(&GV.VarSize),
                               sizeof(GV.VarSize));
      OutBC << llvm::StringRef(reinterpret_cast<const char *>(&GV.DevAddr),
                               sizeof(GV.DevAddr));
      OutBC << llvm::StringRef(reinterpret_cast<const char *>(HostData.get()),
                               GV.VarSize);
      if (CapturedGlobals)
        (*CapturedGlobals)[VarName] =
            std::vector<uint8_t>(HostData.get(), HostData.get() + GV.VarSize);
    }

    size_t TotalBlobs = DeviceMemory.size();
    LOG_DEBUG("Number of Memory Blobs in snapshot:{} stored at position:{}",
              TotalBlobs, OutBC.tell());

    OutBC << llvm::StringRef(reinterpret_cast<const char *>(&TotalBlobs),
                             sizeof(size_t));

    // Write the Device Memory
    for (auto &[Ptr, Blob] : DeviceMemory)
      OutBC << Blob;
    // Lastly write the arguments
    size_t NumArgs = KernelArgSizes.size();
    LOG_DEBUG("Number of Kernel Arguments in snapshot:{} stored at position:{}",
              NumArgs, OutBC.tell());

    OutBC << llvm::StringRef(reinterpret_cast<const char *>(&NumArgs),
                             sizeof(NumArgs));

    for (int I = 0; I < NumArgs; I++) {
      OutBC << llvm::StringRef(
          reinterpret_cast<const char *>(&KernelArgSizes[I]), sizeof(size_t));
      OutBC << llvm::StringRef(reinterpret_cast<const char *>(Args[I]),
                               KernelArgSizes[I]);
    }

    return std::filesystem::canonical(Filename);
  }

  std::filesystem::path static takeMnemeDiffSnapshot(
      std::unordered_map<std::string, proteus::GlobalVarInfo> &GlobalVars,
      llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &DeviceMemory,
      std::filesystem::path &Filename,
      const GlobalSnapshotData &PrologueGlobals, DeviceStream_t Stream) {
    LOG_DEBUG("Storing mneme diff snapshot: {}", Filename.string());
    std::error_code EC;
    auto DEC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
        DeviceTraits<VendorTypes>::DeviceStreamSynchronize(Stream));
    if (DEC)
      LOG_FATAL("Synchronizing stream failed before diff snapshot");

    llvm::raw_fd_ostream OutBC(Filename.string(), EC);
    if (EC)
      LOG_FATAL("Cannot write Mneme diff snapshot: " + EC.message());

    util::writeBytes(OutBC, DiffMagic, DiffMagicSize);

    size_t TotalGlobals = GlobalVars.size();
    util::writeScalar(OutBC, TotalGlobals);
    for (const auto &[VarName, GV] : GlobalVars) {
      auto BaseIt = PrologueGlobals.find(VarName);
      if (BaseIt == PrologueGlobals.end())
        LOG_FATAL("Cannot diff global missing from prologue: " + VarName);
      if (BaseIt->second.size() != GV.VarSize)
        LOG_FATAL("Cannot diff global with size mismatch: " + VarName);

      std::vector<uint8_t> Current(GV.VarSize);
      auto DEC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
          DeviceTraits<VendorTypes>::DeviceCopy(
              Current.data(), const_cast<void *>(GV.DevAddr), GV.VarSize,
              DeviceTraits<VendorTypes>::MemcpyDeviceToHostKind()));
      if (DEC)
        LOG_FATAL("Copying from device to host for global diff failed\n");

      size_t StrLen = VarName.size();
      util::writeScalar(OutBC, StrLen);
      util::writeBytes(OutBC, VarName.data(), StrLen);
      util::writeScalar(OutBC, GV.VarSize);
      util::writeScalar(OutBC, GV.DevAddr);
      size_t NumRanges =
          countChangedRanges(BaseIt->second.data(), Current.data(), GV.VarSize);
      util::writeScalar(OutBC, NumRanges);
      writeChangedRanges(OutBC, BaseIt->second.data(), Current.data(),
                         GV.VarSize, 0);
    }

    size_t TotalBlobs = DeviceMemory.size();
    util::writeScalar(OutBC, TotalBlobs);
    for (auto &[Ptr, Blob] : DeviceMemory) {
      util::writeScalar(OutBC, Blob.getActualSize());
      util::writeScalar(OutBC, Blob.getSize());
      auto *BlobAddr = Blob.getBlobAddr();
      util::writeScalar(OutBC, BlobAddr);
      auto MD = Blob.getMetadata();
      mneme::metadata::serialize(OutBC, MD);

      writeCountAndWriteChangedRanges(OutBC, Blob);
    }

    return std::filesystem::canonical(Filename);
  }

  std::filesystem::path static takeMnemeSnapshot(
      std::unordered_map<std::string, proteus::GlobalVarInfo> &GlobalVars,
      llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &DeviceMemory,
      std::filesystem::path &Filename,
      llvm::SmallVector<size_t> &KernelArgSizes, void **Args,
      DeviceStream_t Stream, GlobalSnapshotData *CapturedGlobals = nullptr,
      SnapshotType Type = SnapshotType::Bytes) {
    if (Type == SnapshotType::Bytes) {
      return takeMnemeBytesSnapshot(GlobalVars, DeviceMemory, Filename,
                                   KernelArgSizes, Args, Stream, CapturedGlobals);
    } else {
      return takeMnemeDiffSnapshot(GlobalVars, DeviceMemory, Filename,
                                  *CapturedGlobals, Stream);
    }
  }

  void static readMnemeSnapShot(
      std::string Filename,
      std::unordered_map<std::string, ReplayGlobalVar> &GlobalVars,
      llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &DeviceMemory,
      std::shared_ptr<KernelInfo> KInfo,
      std::optional<std::string> BaseSnapshotName = std::nullopt) {
    if (!std::filesystem::exists(Filename))
      LOG_FATAL("Mneme Snapshot file does not exist");

    LOG_DEBUG("Opening Snapshot file {}", Filename);

    std::error_code EC;
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> bufferOrErr =
        llvm::MemoryBuffer::getFile(Filename);
    if (std::error_code ec = bufferOrErr.getError())
      LOG_FATAL("Error when opening file " + ec.message());

    // Get a pointer to the raw data in the MemoryBuffer
    llvm::MemoryBuffer *Buffer = bufferOrErr.get().get();
    if (isDiffBuffer(Buffer->getBuffer())) {
      if (!BaseSnapshotName || BaseSnapshotName->empty())
        LOG_FATAL("Mneme diff epilogue requires an explicit base prologue "
                  "snapshot path");
      readDiffMnemeSnapShot(Filename, *BaseSnapshotName, GlobalVars,
                            DeviceMemory, KInfo, Buffer);
      return;
    }

    readFullMnemeSnapShot(Buffer, GlobalVars, DeviceMemory, KInfo);
  }
};

struct KernelInstance {
  std::string PrologueFn;
  std::string EpilogueFn;
  dim3 BlockDim;
  dim3 GridDim;
  llvm::SmallVector<double> ArgValues;
  int NumOccurrences;
  uint64_t SharedMem;
  static llvm::json::Object toJSON(const dim3 &Dim) {
    llvm::json::Object JSONDim;
    JSONDim["x"] = Dim.x;
    JSONDim["y"] = Dim.y;
    JSONDim["z"] = Dim.z;
    return JSONDim;
  }
  llvm::json::Object toJSON() const {
    llvm::json::Object instance;
    instance["Prologue"] = PrologueFn;
    instance["Epilogue"] = EpilogueFn;
    instance["BlockDims"] = KernelInstance::toJSON(BlockDim);
    instance["GridDims"] = KernelInstance::toJSON(GridDim);
    instance["SharedMem"] = SharedMem;
    instance["Args"] = llvm::json::Array(ArgValues);
    instance["Occurrences"] = NumOccurrences;
    return instance;
  }
  KernelInstance(dim3 &GridDim, dim3 &BlockDim, uint64_t SharedMem, void **Args)
      : GridDim(GridDim), BlockDim(BlockDim), SharedMem(SharedMem),
        NumOccurrences(1) {}
  KernelInstance() = default;
};

class KernelInstancesCollection {
  void *VAddr;
  uint64_t VASize;
  llvm::DenseMap<uint64_t, KernelInstance> Instances;
  uint64_t NumRecords;
  int MaxRecordings;
  llvm::SmallVector<size_t> KernelArgSizes;
  llvm::SmallVector<std::string> KernelArgNames;
  llvm::SmallVector<bool> KernelSpecializations;
  llvm::SmallVector<std::function<double(void *)>> ConvertArgToDouble;
  llvm::SmallVector<std::string> ModuleFiles;
  const std::string KName;

private:
  std::string StoreModule(llvm::Module &M, const std::string &RecordReplayDir,
                          uint64_t StaticHash) {
    std::string Filename(
        std::filesystem::path(llvm::Twine(RecordReplayDir + "/RecordedIR_" +
                                          std::to_string(StaticHash) + ".bc")
                                  .str())
            .string());

    std::error_code EC;
    llvm::raw_fd_ostream OutBC(Filename, EC);
    llvm::WriteBitcodeToFile(M, OutBC);
    if (EC)
      LOG_FATAL("Cannot write module ir file");

    LOG_DEBUG("Stored Blob with StaticHash:{} to file {}", StaticHash,
              std::filesystem::canonical(Filename).string());
    OutBC.close();
    return std::filesystem::canonical(Filename).string();
  }

public:
  llvm::json::Object toJSON(uint64_t StaticHash) const {
    llvm::json::Object Collection;
    Collection["StaticHash"] = StaticHash;
    Collection["VAddr"] =
        util::pointerToHexString(reinterpret_cast<uint8_t *>(VAddr));
    Collection["VASize"] = VASize;
    Collection["KernelName"] = KName;
    std::size_t pos = KName.find("__intern__");
    std::string Orig =
        (pos != std::string::npos) ? KName.substr(0, pos) : KName;
    Collection["DemangledName"] = llvm::demangle(Orig);
    Collection["Modules"] = llvm::json::Array(ModuleFiles);
    Collection["BinaryBlobs"] = llvm::json::Array();
    Collection["ArgNames"] = llvm::json::Array(KernelArgNames);
    Collection["Specializations"] = llvm::json::Array(KernelSpecializations);
    llvm::json::Object JSONInstances;
    for (auto &[hash, KI] : Instances) {
      JSONInstances[std::to_string(hash)] = KI.toJSON();
    }
    Collection["instances"] = std::move(JSONInstances);
    return Collection;
  }

  KernelInstancesCollection(const std::string &MnemeDirectory, void *VAddr,
                            uint64_t VASize, proteus::JITKernelInfo &KInfo,
                            int MaxRecordings)
      : VAddr(VAddr), VASize(VASize), MaxRecordings(MaxRecordings),
        NumRecords(0), KName(KInfo.getName()) {
    auto &Module = KInfo.getModule();
    auto *F = Module.getFunction(KInfo.getName());
    KernelArgSizes = mneme::getFuncDescr(*F);
    KernelArgNames = mneme::getArgNames(*F);
    KernelSpecializations = mneme::canSpecialize(*F);
    ConvertArgToDouble = mneme::convertToDouble(*F);
    ModuleFiles.emplace_back(
        StoreModule(Module, MnemeDirectory, KInfo.getStaticHash().getValue()));
  }

  llvm::stable_hash computeHash(dim3 &GridDim, dim3 &BlockDim,
                                uint64_t SharedMem, void **Args) {
    auto BlockHash = llvm::stable_hash_combine((llvm::stable_hash)BlockDim.x,
                                               (llvm::stable_hash)BlockDim.y,
                                               (llvm::stable_hash)BlockDim.z);
    auto GridHash = llvm::stable_hash_combine((llvm::stable_hash)GridDim.x,
                                              (llvm::stable_hash)GridDim.y,
                                              (llvm::stable_hash)GridDim.z);
    auto DHash = llvm::stable_hash_combine(GridHash, BlockHash, SharedMem);
    return DHash;
  }

  template <DeviceVendors VendorTypes>
  std::optional<std::function<
      void(std::unordered_map<std::string, proteus::GlobalVarInfo> &,
           llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &, void **,
           typename DeviceTraits<VendorTypes>::DeviceStream_t)>>
  takeSnapshot(
      std::filesystem::path &MnemeDir,
      std::unordered_map<std::string, proteus::GlobalVarInfo> &GlobalVars,
      llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &DeviceMemory,
      dim3 &GridDim, dim3 &BlockDim, void **Args, size_t SharedMem,
      typename DeviceTraits<VendorTypes>::DeviceStream_t Stream,
      uint64_t StaticHash, typename MnemeSnapshot<VendorTypes>::SnapshotType epilogueSnapshotType) {

    if (NumRecords >= MaxRecordings)
      return std::nullopt;

    auto DynamicHash = computeHash(GridDim, BlockDim, SharedMem, Args);

    if (Instances.contains(DynamicHash)) {
      Instances[DynamicHash].NumOccurrences++;
      LOG_DEBUG(
          "Kernel {} with DynamicHash {} is already recorded, skipping ...",
          StaticHash, DynamicHash);
      return std::nullopt;
    }

    NumRecords++;

    LOG_DEBUG("First Instance of Kernel {} with DynamicHash {}, recording ...",
              StaticHash, DynamicHash);

    Instances.insert(
        {DynamicHash, KernelInstance(GridDim, BlockDim, SharedMem, Args)});
    std::filesystem::path Filename(MnemeDir /
                                   (std::string("DeviceState.prologue.") +
                                    std::to_string(StaticHash) + "." +
                                    std::to_string(DynamicHash) + ".mneme"));

    using SnapshotT = MnemeSnapshot<VendorTypes>;
    auto PrologueGlobals =
        std::make_shared<typename SnapshotT::GlobalSnapshotData>();
    auto snapshotType = SnapshotT::SnapshotType::Bytes; // prologues are always Bytes
    Instances[DynamicHash].PrologueFn =
        SnapshotT::takeMnemeSnapshot(GlobalVars, DeviceMemory, Filename,
                                     KernelArgSizes, Args, Stream,
                                     PrologueGlobals.get(), snapshotType)
            .string();

    
    std::function<void(
        std::unordered_map<std::string, proteus::GlobalVarInfo> &,
        llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &, void **,
        typename DeviceTraits<VendorTypes>::DeviceStream_t)>
        CaptureEpilogue =
            [this, DynamicHash, StaticHash, MnemeDir, PrologueGlobals, epilogueSnapshotType](
                std::unordered_map<std::string, proteus::GlobalVarInfo>
                    &GlobalVars,
                llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>>
                    &DeviceMemory,
                void **Args,
                typename DeviceTraits<VendorTypes>::DeviceStream_t Stream) {
              std::filesystem::path Filename(
                  MnemeDir / (std::string("DeviceState.epilogue.") +
                              std::to_string(StaticHash) + "." +
                              std::to_string(DynamicHash) + ".mneme"));

              Instances[DynamicHash].EpilogueFn =
                  SnapshotT::takeMnemeSnapshot(
                      GlobalVars, DeviceMemory, Filename, KernelArgSizes, Args, 
                      Stream, PrologueGlobals.get(), epilogueSnapshotType)
                      .string();
            };
    return CaptureEpilogue;
  }
};

class RecordDatabase {
  std::filesystem::path MnemeDirectory;
  std::regex KernelWhiteList;
  std::string RegexStr;
  bool HasRegex;
  llvm::DenseMap<uint64_t, KernelInstancesCollection> KernelRecords;
  uint64_t MaxRecordings;
  std::string epilogueSnapshotType;

public:
  RecordDatabase() : KernelWhiteList(""), HasRegex(false), epilogueSnapshotType("bytes") {
    auto WhiteList = std::getenv("MNEME_RR_KERNELS");
    if (WhiteList) {
      HasRegex = true;
      RegexStr = std::string(WhiteList);
      KernelWhiteList = std::string(WhiteList);
    }

    auto Dir = std::getenv("MNEME_DATA_DIR");
    MnemeDirectory =
        (Dir ? std::string(Dir) : std::filesystem::current_path().string());

    if (!std::filesystem::is_directory(MnemeDirectory)) {
      throw std::runtime_error("Path :" + MnemeDirectory.string() +
                               " does not exist.\n");
    }
    MnemeDirectory = std::filesystem::absolute(MnemeDirectory);
    MaxRecordings = 4;
    auto UMaxRecordings = std::getenv("MNEME_MAX_RECORDINGS");
    if (UMaxRecordings) {
      MaxRecordings = std::atoi(UMaxRecordings);
    }

    auto EpilogueSnapshotTypeEnv = std::getenv("MNEME_EPILOGUE_TYPE");
    if (EpilogueSnapshotTypeEnv) {
      epilogueSnapshotType = std::string(EpilogueSnapshotTypeEnv);
    }

  }

  ~RecordDatabase() {
    for (auto &[StaticHash, Record] : KernelRecords) {
      auto JsonFilename =
          MnemeDirectory / (std::to_string(StaticHash) + ".json");
      std::error_code EC;
      auto JSONRecord = Record.toJSON(StaticHash);
      llvm::raw_fd_ostream JsonOS(JsonFilename.string(), EC);
      JsonOS << llvm::json::Value(std::move(JSONRecord));
      JsonOS.close();
    }
  }

  bool shouldRecord(const std::string &KernelName) const {
    if (!HasRegex)
      return true;

    try {
      return std::regex_search(KernelName, KernelWhiteList) ||
             std::regex_search(llvm::demangle(KernelName), KernelWhiteList);
    } catch (const std::regex_error &e) {
      LOG_WARN("Invalid regex: {}, ... falling back and recording everything");
    }
    return true;
  }

  template <DeviceVendors VendorTypes>
  typename MnemeSnapshot<VendorTypes>::SnapshotType parseSnapshotType(const std::string &TypeStr) {
    if (TypeStr == "bytes")
      return MnemeSnapshot<VendorTypes>::SnapshotType::Bytes;
    else if (TypeStr == "diff")
      return MnemeSnapshot<VendorTypes>::SnapshotType::Diff;
    else
      LOG_FATAL("Invalid snapshot type: " + TypeStr);
    return MnemeSnapshot<VendorTypes>::SnapshotType::Bytes;
  }

  template <DeviceVendors VendorTypes>
  std::optional<std::function<
      void(std::unordered_map<std::string, proteus::GlobalVarInfo> &,
           llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &, void **,
           typename DeviceTraits<VendorTypes>::DeviceStream_t)>>
  takeSnapshot(
      void *VAddr, uint64_t VASize, proteus::JITKernelInfo &KInfo,
      llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &DeviceMemory,
      dim3 &GridDim, dim3 &BlockDim, void **Args, size_t SharedMem,
      typename DeviceTraits<VendorTypes>::DeviceStream_t Stream) {
    using namespace proteus;
    using SnapshotT = MnemeSnapshot<VendorTypes>::SnapshotType;

    if (!shouldRecord(KInfo.getName())) {
      LOG_INFO("Skip record of Kernel");
      return std::nullopt;
    }

    // how do we want to save the epilogue; parse here once we have vendor type
    const SnapshotT epilogueType = parseSnapshotType<VendorTypes>(epilogueSnapshotType);

    auto IT = KernelRecords.try_emplace(
        KInfo.getStaticHash().getValue(),
        KernelInstancesCollection(getDir(), VAddr, VASize, KInfo,
                                  MaxRecordings));
    LOG_INFO("Created instance");
    return IT.first->second.takeSnapshot<VendorTypes>(
        MnemeDirectory, KInfo.getBinaryInfo().getVarNameToGlobalInfo(),
        DeviceMemory, GridDim, BlockDim, Args, SharedMem, Stream,
        KInfo.getStaticHash().getValue(), epilogueType);
  }

  const std::string getDir() const { return MnemeDirectory.string(); }
};

} // namespace mneme
