#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MD5.h>
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
  std::string SourceFile;
  std::string SourceCopy;
  std::string SourceMD5;

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

    // Only an absolute path identifies the translation unit. LTO-linked
    // modules carry a placeholder name and are reported as unknown.
    std::string ModuleSource = Mod->getSourceFileName();
    if (std::filesystem::path(ModuleSource).is_absolute())
      SourceFile = ModuleSource;
    else
      LOG_DEBUG("No translation unit source for kernel {} (module source '{}')",
                KName, ModuleSource);
  }

  // Copy the translation unit source next to the recorded IR, named by content
  // hash so kernels from the same file share one copy.
  void copySourceFile(const std::string &RecordReplayDir) {
    auto BufOrErr = llvm::MemoryBuffer::getFile(SourceFile);
    if (!BufOrErr) {
      LOG_WARN("Cannot read source file {} for kernel {}: {}", SourceFile,
               KName, BufOrErr.getError().message());
      return;
    }
    llvm::StringRef Contents = (*BufOrErr)->getBuffer();

    llvm::MD5 Hash;
    Hash.update(Contents);
    llvm::MD5::MD5Result Result;
    Hash.final(Result);
    std::string Digest = Result.digest().str().str();

    std::string Basename =
        std::filesystem::path(SourceFile).filename().string();
    std::string Filename = "RecordedSource_" + Digest + "_" + Basename;
    std::filesystem::path Dest =
        std::filesystem::path(RecordReplayDir) / Filename;
    if (!std::filesystem::exists(Dest)) {
      std::error_code EC;
      llvm::raw_fd_ostream Out(Dest.string(), EC);
      if (EC) {
        LOG_WARN("Cannot write source copy {}: {}", Dest.string(),
                 EC.message());
        return;
      }
      Out << Contents;
    }

    SourceCopy = Filename;
    SourceMD5 = Digest;
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
    if (!SourceFile.empty())
      Collection["SourceFile"] = SourceFile;
    if (!SourceCopy.empty()) {
      Collection["SourceCopy"] = SourceCopy;
      Collection["SourceMD5"] = SourceMD5;
    }
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
                            int MaxRecordings, bool CopySource)
      : VAddr(VAddr), VASize(VASize), MaxRecordings(MaxRecordings),
        NumRecords(0), KName(KInfo.getName()) {
    const auto &BitcodeBytes = KInfo.getBitcode();
    llvm::StringRef Bitcode(BitcodeBytes.data(), BitcodeBytes.size());
    if (Bitcode.empty())
      LOG_FATAL("Empty bitcode for kernel " + KName);

    extractArgInfoFromBitcode(Bitcode);
    if (CopySource && !SourceFile.empty())
      copySourceFile(MnemeDirectory);
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

    auto PrologueGlobals = std::make_shared<GlobalSnapshotData>();
    SnapshotInput<VendorTypes> In{GlobalVars, DeviceMemory, KernelArgSizes,
                                  Args, Stream};
    Instances[DynamicHash].PrologueFn =
        BytesWriter<VendorTypes>(PrologueGlobals).write(Filename, In).string();

    // std::function requires a copyable callable, so the writer is shared.
    std::shared_ptr<SnapshotWriter<VendorTypes>> Writer =
        makeEpilogueWriter<VendorTypes>(EpilogueType, PrologueGlobals);

    std::function<void(llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &,
                       void **,
                       typename DeviceTraits<VendorTypes>::DeviceStream_t)>
        CaptureEpilogue =
            [this, DynamicHash, StaticHash, MnemeDir, GlobalVars, Writer](
                llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>>
                    &DeviceMemory,
                void **Args,
                typename DeviceTraits<VendorTypes>::DeviceStream_t Stream) {
              std::filesystem::path Filename(
                  MnemeDir / (std::string("DeviceState.epilogue.") +
                              std::to_string(StaticHash) + "." +
                              std::to_string(DynamicHash) + ".mneme"));

              SnapshotInput<VendorTypes> In{GlobalVars, DeviceMemory,
                                            KernelArgSizes, Args, Stream};
              Instances[DynamicHash].EpilogueFn =
                  Writer->write(Filename, In).string();
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
  bool CopySource;

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
    CopySource = Conf.CopySource;
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
                                              MaxRecordings, CopySource));
    LOG_INFO("Created instance");
    return IT.first->second.takeSnapshot<VendorTypes>(
        MnemeDirectory, KInfo.getGlobals(), DeviceMemory, GridDim, BlockDim,
        Args, SharedMem, Stream, StaticHash, EpilogueType);
  }

  const std::string getDir() const { return MnemeDirectory.string(); }
};

} // namespace mneme
