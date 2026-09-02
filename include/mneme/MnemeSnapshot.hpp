#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <llvm/ADT/ArrayRef.h>
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
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <string>
#include <sys/types.h>
#include <type_traits>

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeConfig.hpp"
#include "mneme/MnemeKernelInfo.hpp"
#include "mneme/MnemeLLVMUtils.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeMemory.hpp"
#include "mneme/MnemeSnapshotFormat.hpp"
#include "mneme/MnemeSnapshotRecords.hpp"
#include "mneme/MnemeUtils.hpp"
#include <proteus/KernelMetadata.h>

namespace mneme {

template <DeviceVendors VendorTypes> class MnemeSnapshot {
  using MnemeDeviceRT = DeviceTraits<VendorTypes>;
  using DeviceError_t = typename MnemeDeviceRT::DeviceError_t;
  using DeviceStream_t = typename MnemeDeviceRT::DeviceStream_t;
  using KernelFunction_t = typename MnemeDeviceRT::KernelFunction_t;
  static constexpr size_t DiffChunkSize = 1 << 20;

  class CountingRawOStream : public llvm::raw_ostream {
    uint64_t Pos = 0;

    void write_impl(const char *, size_t Size) override { Pos += Size; }
    uint64_t current_pos() const override { return Pos; }

  public:
    CountingRawOStream() : llvm::raw_ostream(/*unbuffered=*/true) {}
    uint64_t bytesWritten() const { return tell(); }
  };

  static size_t countChangedRanges(llvm::ArrayRef<uint8_t> Base,
                                   llvm::ArrayRef<uint8_t> Current) {
    if (Base.size() != Current.size())
      LOG_FATAL("Cannot diff buffers with different sizes");

    // Count the number of contiguous ranges that have changed between Base and
    // Current. We want to write out the number of ranges so that the reader 
    // can know how many ranges to read.
    size_t Count = 0;
    bool InRange = false;
    for (size_t I = 0; I < Base.size(); ++I) {
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

  static size_t
  writeChangedRanges(llvm::raw_ostream &OS, llvm::ArrayRef<uint8_t> Base,
                     llvm::ArrayRef<uint8_t> Current, size_t BaseOffset,
                     llvm::MutableArrayRef<uint8_t> UpdateBase = {}) {
    if (Base.size() != Current.size())
      LOG_FATAL("Cannot diff buffers with different sizes");
    if (!UpdateBase.empty() && UpdateBase.size() != Base.size())
      LOG_FATAL("Cannot update diff base with mismatched buffer size");

    // Write out the contiguous ranges that have changed between Base 
    // and Current.
    size_t Count = 0;
    size_t I = 0;
    while (I < Base.size()) {
      while (I < Base.size() && Base[I] == Current[I])
        ++I;
      if (I == Base.size())
        break;

      size_t Start = I;
      while (I < Base.size() && Base[I] != Current[I])
        ++I;

      size_t Offset = BaseOffset + Start;
      size_t Len = I - Start;
      util::writeScalar(OS, Offset);
      util::writeScalar(OS, Len);
      util::writeBytes(OS, Current.slice(Start, Len));
      if (!UpdateBase.empty())
        std::memcpy(UpdateBase.data() + Start, Current.data() + Start, Len);
      ++Count;
    }
    return Count;
  }

  static void writeCountAndWriteChangedRanges(
      llvm::raw_ostream &OS, MnemeMemoryBlob<VendorTypes> &Blob,
      bool UpdateBaseData = true) {
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

      llvm::ArrayRef<uint8_t> ChunkBase(Base + Offset, ChunkSize);
      llvm::ArrayRef<uint8_t> ChunkCurrent(Scratch.get(), ChunkSize);
      if (UpdateBaseData) {
        llvm::MutableArrayRef<uint8_t> UpdateBase(Base + Offset, ChunkSize);
        NumRanges += writeChangedRanges(DiffOS, ChunkBase, ChunkCurrent, Offset,
                                        UpdateBase);
      } else {
        NumRanges +=
            writeChangedRanges(DiffOS, ChunkBase, ChunkCurrent, Offset);
      }
    }

    util::writeScalar(OS, NumRanges);
    util::writeBytes(OS, llvm::StringRef(DiffBytes.data(), DiffBytes.size()));
  }

  static size_t computeMnemeBytesSnapshotSize(
      const proteus::runtime::GlobalMetadataMap &GlobalVars,
      const llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &DeviceMemory,
      llvm::ArrayRef<size_t> KernelArgSizes) {
    size_t Size = sizeof(size_t);
    for (const auto &[VarName, GV] : GlobalVars) {
      Size +=
          GlobalVarHeader{VarName, GV.VarSize, const_cast<void *>(GV.DevAddr)}
              .serializedSize();
      Size += GV.VarSize;
    }

    Size += sizeof(size_t);
    for (const auto &[Ptr, Blob] : DeviceMemory) {
      Size += BlobHeader::serializedSize();
      Size += Blob.getSize();
      Size += metadata::serializedSize(Blob.getMetadata());
    }

    Size += sizeof(size_t);
    for (size_t ArgSize : KernelArgSizes) {
      Size += sizeof(size_t);
      Size += ArgSize;
    }
    return Size;
  }

public:
  using GlobalSnapshotData = mneme::GlobalSnapshotData;

  std::filesystem::path static takeMnemeBytesSnapshot(
      const proteus::runtime::GlobalMetadataMap &GlobalVars,
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

      GlobalVarHeader{VarName, GV.VarSize, const_cast<void *>(GV.DevAddr)}
          .write(OutBC);
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

    return Filename.filename();
  }

  static void writeMnemeDiffSnapshot(
      llvm::raw_ostream &OutBC,
      const proteus::runtime::GlobalMetadataMap &GlobalVars,
      llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &DeviceMemory,
      const GlobalSnapshotData &PrologueGlobals, bool UpdateBaseData) {
    SnapshotHeader{SnapshotKind::Diff, 1}.write(OutBC);

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

      GlobalVarHeader{VarName, GV.VarSize, const_cast<void *>(GV.DevAddr)}
          .write(OutBC);
      size_t NumRanges = countChangedRanges(BaseIt->second, Current);
      util::writeScalar(OutBC, NumRanges);
      writeChangedRanges(OutBC, BaseIt->second, Current, 0);
    }

    size_t TotalBlobs = DeviceMemory.size();
    util::writeScalar(OutBC, TotalBlobs);
    for (auto &[Ptr, Blob] : DeviceMemory) {
      BlobHeader{Blob.getActualSize(), Blob.getSize(), Blob.getBlobAddr()}
          .write(OutBC);
      auto MD = Blob.getMetadata();
      mneme::metadata::serialize(OutBC, MD);

      writeCountAndWriteChangedRanges(OutBC, Blob, UpdateBaseData);
    }
  }

  static size_t measureMnemeDiffSnapshotSize(
      const proteus::runtime::GlobalMetadataMap &GlobalVars,
      llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &DeviceMemory,
      const GlobalSnapshotData &PrologueGlobals, DeviceStream_t Stream) {
    auto DEC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
        DeviceTraits<VendorTypes>::DeviceStreamSynchronize(Stream));
    if (DEC)
      LOG_FATAL("Synchronizing stream failed before diff snapshot");

    CountingRawOStream Counter;
    writeMnemeDiffSnapshot(Counter, GlobalVars, DeviceMemory, PrologueGlobals,
                           false);
    return Counter.bytesWritten();
  }

  std::filesystem::path static takeMnemeDiffSnapshot(
      const proteus::runtime::GlobalMetadataMap &GlobalVars,
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

    writeMnemeDiffSnapshot(OutBC, GlobalVars, DeviceMemory, PrologueGlobals,
                           true);
    return Filename.filename();
  }

  std::filesystem::path static takeBestMnemeSnapshot(
      const proteus::runtime::GlobalMetadataMap &GlobalVars,
      llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &DeviceMemory,
      std::filesystem::path &Filename,
      llvm::SmallVector<size_t> &KernelArgSizes, void **Args,
      const GlobalSnapshotData &PrologueGlobals, DeviceStream_t Stream) {
    size_t BytesSize =
        computeMnemeBytesSnapshotSize(GlobalVars, DeviceMemory, KernelArgSizes);
    size_t DiffSize = measureMnemeDiffSnapshotSize(GlobalVars, DeviceMemory,
                                                   PrologueGlobals, Stream);

    if (DiffSize <= BytesSize)
      return takeMnemeDiffSnapshot(GlobalVars, DeviceMemory, Filename,
                                   PrologueGlobals, Stream);

    return takeMnemeBytesSnapshot(GlobalVars, DeviceMemory, Filename,
                                  KernelArgSizes, Args, Stream);
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
  // Parse Proteus's serialized bitcode in a Mneme-owned LLVMContext and
  // extract per-argument metadata. Operating on a Mneme-owned Module 
  // keeps Mneme's LLVM runtime from touching any
  // Proteus-owned LLVM C++ object across the DSO boundary.
  void extractArgInfoFromBitcode(llvm::StringRef Bitcode) {
    auto Ctx = std::make_unique<llvm::LLVMContext>();
    llvm::MemoryBufferRef BufRef(Bitcode, KName);
    auto ModOrErr = llvm::parseBitcodeFile(BufRef, *Ctx);
    if (!ModOrErr)
      LOG_FATAL("parseBitcodeFile failed for kernel " + KName + ": " +
                llvm::toString(ModOrErr.takeError()));
    std::unique_ptr<llvm::Module> Mod = std::move(*ModOrErr);

    llvm::Function *F = Mod->getFunction(KName);
    if (!F)
      LOG_FATAL("Function " + KName + " not found in parsed bitcode");

    KernelArgSizes = mneme::getFuncDescr(*F);
    KernelArgNames = mneme::getArgNames(*F);
    KernelSpecializations = mneme::canSpecialize(*F);
    ConvertArgToDouble = mneme::convertToDouble(*F);
  }

  std::string StoreModuleBytes(llvm::StringRef Bytes,
                               const std::string &RecordReplayDir,
                               uint64_t StaticHash) {
    std::string Filename(
        std::filesystem::path(llvm::Twine(RecordReplayDir + "/RecordedIR_" +
                                          std::to_string(StaticHash) + ".bc")
                                  .str())
            .string());

    std::error_code EC;
    llvm::raw_fd_ostream OutBC(Filename, EC);
    if (EC)
      LOG_FATAL("Cannot write module ir file");
    OutBC << Bytes;
    OutBC.close();

    LOG_DEBUG("Stored Blob with StaticHash:{} to file {}", StaticHash,
              std::filesystem::canonical(Filename).string());
    return std::filesystem::path(Filename).filename().string();
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
                            uint64_t VASize,
                            const proteus::runtime::KernelMetadata &KInfo,
                            int MaxRecordings)
      : VAddr(VAddr), VASize(VASize), MaxRecordings(MaxRecordings),
        NumRecords(0), KName(KInfo.getName()) {
    const auto &BitcodeBytes = KInfo.getBitcode();
    llvm::StringRef Bitcode(BitcodeBytes.data(), BitcodeBytes.size());
    if (Bitcode.empty())
      LOG_FATAL("Empty bitcode for kernel " + KName);

    extractArgInfoFromBitcode(Bitcode);
    ModuleFiles.emplace_back(StoreModuleBytes(
        Bitcode, MnemeDirectory, KInfo.getStaticHash()));
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
      void(llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &, void **,
           typename DeviceTraits<VendorTypes>::DeviceStream_t)>>
  takeSnapshot(
      std::filesystem::path &MnemeDir,
      const proteus::runtime::GlobalMetadataMap &GlobalVars,
      llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &DeviceMemory,
      dim3 &GridDim, dim3 &BlockDim, void **Args, size_t SharedMem,
      typename DeviceTraits<VendorTypes>::DeviceStream_t Stream,
      uint64_t StaticHash, EpilogueSnapshotType EpilogueType) {

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
    Instances[DynamicHash].PrologueFn =
        SnapshotT::takeMnemeBytesSnapshot(GlobalVars, DeviceMemory, Filename,
                                          KernelArgSizes, Args, Stream,
                                          PrologueGlobals.get())
            .string();

    std::function<void(llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &,
                       void **,
                       typename DeviceTraits<VendorTypes>::DeviceStream_t)>
        CaptureEpilogue =
            [this, DynamicHash, StaticHash, MnemeDir, PrologueGlobals,
             GlobalVars, EpilogueType](
                llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>>
                    &DeviceMemory,
                void **Args,
                typename DeviceTraits<VendorTypes>::DeviceStream_t Stream) {
              std::filesystem::path Filename(
                  MnemeDir / (std::string("DeviceState.epilogue.") +
                              std::to_string(StaticHash) + "." +
                              std::to_string(DynamicHash) + ".mneme"));

              switch (EpilogueType) {
              case EpilogueSnapshotType::Bytes:
                Instances[DynamicHash].EpilogueFn =
                    SnapshotT::takeMnemeBytesSnapshot(
                        GlobalVars, DeviceMemory, Filename, KernelArgSizes,
                        Args, Stream)
                        .string();
                break;
              case EpilogueSnapshotType::Diff:
                Instances[DynamicHash].EpilogueFn =
                    SnapshotT::takeMnemeDiffSnapshot(
                        GlobalVars, DeviceMemory, Filename, *PrologueGlobals,
                        Stream)
                        .string();
                break;
              case EpilogueSnapshotType::Best:
                Instances[DynamicHash].EpilogueFn =
                    SnapshotT::takeBestMnemeSnapshot(
                        GlobalVars, DeviceMemory, Filename, KernelArgSizes,
                        Args, *PrologueGlobals, Stream)
                        .string();
                break;
              }
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
  llvm::DenseMap<uint64_t, uint64_t> KernelLaunchCounts;
  uint64_t MaxRecordings;
  uint64_t SkipRecordings;
  EpilogueSnapshotType EpilogueType;

public:
  RecordDatabase() : KernelWhiteList(""), HasRegex(false) {
    const auto &Conf = Config::get();
    if (Conf.KernelRegex) {
      HasRegex = true;
      RegexStr = *Conf.KernelRegex;
      KernelWhiteList = RegexStr;
    }

    MnemeDirectory = Conf.getDataDirectory();
    MaxRecordings = Conf.MaxRecordings;
    SkipRecordings = Conf.SkipRecordings;
    EpilogueType = Conf.EpilogueType;
  }

  void writeKernelJSON(uint64_t StaticHash) {
    auto It = KernelRecords.find(StaticHash);
    if (It == KernelRecords.end()) {
      LOG_WARN("Attempted to write JSON for unrecorded kernel hash {}", StaticHash);
      return;
    }
    auto* RecordPtr = &It->second;

    auto JsonFilename = MnemeDirectory / (std::to_string(StaticHash) + ".json");
    auto JSONRecord = RecordPtr->toJSON(StaticHash);

    std::error_code EC;
    llvm::raw_fd_ostream JsonOS(JsonFilename.string(), EC);
    if (EC) {
      LOG_WARN("Failed to open JSON file for kernel {}: {}", StaticHash, EC.message());
      return;
    }

    JsonOS << llvm::json::Value(std::move(JSONRecord));
    JsonOS.close();
    if (JsonOS.has_error()) {
      LOG_WARN("Failed to write JSON for kernel {}: {}", StaticHash,
               JsonOS.error().message());
      return;
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
  std::optional<std::function<
      void(llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &, void **,
           typename DeviceTraits<VendorTypes>::DeviceStream_t)>>
  takeSnapshot(
      void *VAddr, uint64_t VASize,
      const proteus::runtime::KernelMetadata &KInfo,
      llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &DeviceMemory,
      dim3 &GridDim, dim3 &BlockDim, void **Args, size_t SharedMem,
      typename DeviceTraits<VendorTypes>::DeviceStream_t Stream) {
    if (!shouldRecord(KInfo.getName())) {
      LOG_INFO("Skip record of Kernel");
      return std::nullopt;
    }

    auto StaticHash = KInfo.getStaticHash();
    uint64_t &LaunchCount = KernelLaunchCounts[StaticHash];
    LaunchCount++;
    if (LaunchCount <= SkipRecordings) {
      LOG_DEBUG("Skipping recording {} of {} for kernel {}", LaunchCount,
                SkipRecordings, KInfo.getName());
      return std::nullopt;
    }

    auto IT = KernelRecords.try_emplace(
        StaticHash, KernelInstancesCollection(getDir(), VAddr, VASize, KInfo,
                                              MaxRecordings));
    LOG_INFO("Created instance");
    return IT.first->second.takeSnapshot<VendorTypes>(
        MnemeDirectory, KInfo.getGlobals(), DeviceMemory, GridDim, BlockDim,
        Args, SharedMem, Stream, StaticHash, EpilogueType);
  }

  const std::string getDir() const { return MnemeDirectory.string(); }
};

} // namespace mneme
