#pragma once
#include <cstdint>
#include <filesystem>
#include <functional>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <optional>
#include <regex>
#include <unordered_map>
#include <unordered_set>

#include "llvm/Demangle/Demangle.h"
#include <llvm/ADT/ArrayRef.h>
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

class CapturedGlobals {
  std::unordered_map<std::string, proteus::GlobalVarInfo> Globals;

public:
  void add(const std::string &Name, const proteus::GlobalVarInfo &GVI) {
    auto [It, Inserted] = Globals.try_emplace(Name, GVI);
    if (!Inserted && (It->second.DevAddr != GVI.DevAddr ||
                      It->second.VarSize != GVI.VarSize))
      LOG_FATAL("Proteus registered global name collision for '" + Name +
                "' with different device storage");
  }

  size_t size() const { return Globals.size(); }

  const std::unordered_map<std::string, proteus::GlobalVarInfo> &items() const {
    return Globals;
  }

  std::optional<std::string> findContaining(const void *Ptr) const {
    std::optional<std::string> Match;
    for (const auto &[Name, GVI] : Globals) {
      DeviceAddressRange Range{GVI.DevAddr, GVI.VarSize};
      if (!Range.contains(Ptr))
        continue;

      if (Match && *Match != Name)
        LOG_FATAL("Device pointer " + util::pointerToHexString(Ptr) +
                  " aliases multiple Proteus globals: " + *Match + " and " +
                  Name);
      Match = Name;
    }
    return Match;
  }
};

template <DeviceVendors VendorTypes> class MnemeSnapshot {
  using MnemeDeviceRT = DeviceTraits<VendorTypes>;
  using DeviceError_t = typename MnemeDeviceRT::DeviceError_t;
  using DeviceStream_t = typename MnemeDeviceRT::DeviceStream_t;
  using KernelFunction_t = typename MnemeDeviceRT::KernelFunction_t;

public:
  static std::pair<std::string, ReplayGlobalVar>
  fromBuffer(const char *&Buffer) {
    const char *tmp = Buffer;
    size_t StrLen = util::extractScalar<size_t>(Buffer);
    std::string Name{Buffer, StrLen};
    Buffer += StrLen;
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

  std::filesystem::path static takeMnemeSnapshot(
      const CapturedGlobals &GlobalVars,
      llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &DeviceMemory,
      std::filesystem::path &Filename,
      llvm::SmallVector<size_t> &KernelArgSizes, void **Args,
      llvm::ArrayRef<KernelArgPointerSlot> PointerSlots,
      DeviceStream_t Stream) {
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

    for (const auto &[VarName, GV] : GlobalVars.items()) {
      std::cout << "Reading " << VarName << " " << GV.HostAddr << " "
                << GV.DevAddr << " " << GV.VarSize << "\n";
      uint8_t *HostData = new uint8_t[GV.VarSize];
      auto DEC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
          DeviceTraits<VendorTypes>::DeviceCopy(
              HostData, const_cast<void *>(GV.DevAddr), GV.VarSize,
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
      OutBC << llvm::StringRef(reinterpret_cast<const char *>(HostData),
                               GV.VarSize);
      delete[] HostData;
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

    uint64_t NumPointerOffsets = PointerSlots.size();
    LOG_DEBUG("Number of pointer offsets in snapshot:{} stored at position:{}",
              NumPointerOffsets, OutBC.tell());
    OutBC << llvm::StringRef(reinterpret_cast<const char *>(&NumPointerOffsets),
                             sizeof(NumPointerOffsets));
    for (auto PointerSlot : PointerSlots) {
      OutBC << llvm::StringRef(
          reinterpret_cast<const char *>(&PointerSlot.ArgIndex),
          sizeof(PointerSlot.ArgIndex));
      OutBC << llvm::StringRef(
          reinterpret_cast<const char *>(&PointerSlot.ByteOffset),
          sizeof(PointerSlot.ByteOffset));
    }

    return std::filesystem::canonical(Filename);
  }

  void static readMnemeSnapShot(
      std::string Filename,
      std::unordered_map<std::string, ReplayGlobalVar> &GlobalVars,
      llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &DeviceMemory,
      std::shared_ptr<KernelInfo> KInfo) {
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
    auto *Start = Buffer->getBufferStart();
    auto *End = Buffer->getBufferEnd();
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

    KInfo->PointerSlots.clear();
    if (CurrentPtr < End) {
      auto TotalPointerOffsets = util::extractScalar<uint64_t>(CurrentPtr);
      LOG_DEBUG("Snapshot contains {} pointer offsets starting at location {}",
                TotalPointerOffsets, (uintptr_t)CurrentPtr - (uintptr_t)Start);
      for (auto I = 0; I < TotalPointerOffsets; ++I) {
        KernelArgPointerSlot PointerSlot;
        PointerSlot.ArgIndex = util::extractScalar<uint64_t>(CurrentPtr);
        PointerSlot.ByteOffset = util::extractScalar<uint64_t>(CurrentPtr);
        KInfo->PointerSlots.emplace_back(PointerSlot);
      }
    }
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
  llvm::SmallVector<KernelArgPointerSlot> PointerSlots;
  std::unordered_set<std::string> CapturedGlobalNames;
  std::unordered_set<std::string> StoredSidecarGlobals;
  const std::string KName;

private:
  std::string StoreModule(llvm::Module &M, const std::string &RecordReplayDir,
                          uint64_t StaticHash, const std::string &Suffix = "") {
    std::string Filename(
        std::filesystem::path(llvm::Twine(RecordReplayDir + "/RecordedIR_" +
                                          std::to_string(StaticHash) + Suffix +
                                          ".bc")
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
    llvm::json::Array Globals;
    for (const auto &Name : CapturedGlobalNames)
      Globals.emplace_back(Name);
    Collection["Globals"] = std::move(Globals);
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

  KernelInstancesCollection(
      const std::string &MnemeDirectory, void *VAddr, uint64_t VASize,
      proteus::JITKernelInfo &KInfo,
      llvm::ArrayRef<KernelArgPointerSlot> KernelPointerSlots,
      int MaxRecordings)
      : VAddr(VAddr), VASize(VASize), NumRecords(0),
        MaxRecordings(MaxRecordings),
        PointerSlots(KernelPointerSlots.begin(), KernelPointerSlots.end()),
        KName(KInfo.getName()) {
    auto &Module = KInfo.getModule();
    auto *F = Module.getFunction(KInfo.getName());
    KernelArgSizes = mneme::getFuncDescr(*F);
    KernelArgNames = mneme::getArgNames(*F);
    KernelSpecializations = mneme::canSpecialize(*F);
    ConvertArgToDouble = mneme::convertToDouble(*F);
    ModuleFiles.emplace_back(
        StoreModule(Module, MnemeDirectory, KInfo.getStaticHash().getValue()));
  }

  void addCapturedGlobalNames(const CapturedGlobals &GlobalVars) {
    for (const auto &[Name, GVI] : GlobalVars.items())
      CapturedGlobalNames.insert(Name);
  }

  template <typename ProteusT>
  void addGlobalSidecarModules(std::filesystem::path &MnemeDir,
                               uint64_t StaticHash,
                               proteus::JITKernelInfo &KInfo, ProteusT &Proteus,
                               const CapturedGlobals &GlobalVars) {
    auto HasDefinition = [](llvm::Module &M, const std::string &Name) {
      auto *GV = M.getGlobalVariable(Name);
      return GV && !GV->isDeclaration();
    };

    auto &KernelModule = KInfo.getModule();
    for (const auto &[Name, GVI] : GlobalVars.items()) {
      if (HasDefinition(KernelModule, Name) || StoredSidecarGlobals.count(Name))
        continue;

      bool Stored = false;
      for (auto &[Handle, BinInfo] : Proteus.HandleToBinaryInfo) {
        if (!BinInfo.hasExtractedModules())
          Proteus.extractModules(BinInfo);

        for (auto ModuleRef : BinInfo.getExtractedModules()) {
          llvm::Module &M = ModuleRef.get();
          if (!HasDefinition(M, Name))
            continue;

          llvm::ValueToValueMapTy VMap;
          auto Sidecar =
              llvm::CloneModule(M, VMap, [&](const llvm::GlobalValue *GV) {
                if (auto *GVar = llvm::dyn_cast<llvm::GlobalVariable>(GV))
                  return GVar->getName() == Name;
                return false;
              });
          ModuleFiles.emplace_back(StoreModule(
              *Sidecar, MnemeDir.string(), StaticHash,
              ".globals." + std::to_string(StoredSidecarGlobals.size())));
          StoredSidecarGlobals.insert(Name);
          Stored = true;
          break;
        }
        if (Stored)
          break;
      }

      if (!Stored)
        LOG_FATAL("Captured global '" + Name +
                  "' was registered by Proteus but no LLVM definition was "
                  "available for replay");
    }
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
      void(const CapturedGlobals &,
           llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &, void **,
           typename DeviceTraits<VendorTypes>::DeviceStream_t)>>
  takeSnapshot(
      std::filesystem::path &MnemeDir, const CapturedGlobals &GlobalVars,
      llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &DeviceMemory,
      dim3 &GridDim, dim3 &BlockDim, void **Args, size_t SharedMem,
      typename DeviceTraits<VendorTypes>::DeviceStream_t Stream,
      uint64_t StaticHash) {

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

    Instances[DynamicHash].PrologueFn =
        MnemeSnapshot<VendorTypes>::takeMnemeSnapshot(
            GlobalVars, DeviceMemory, Filename, KernelArgSizes, Args,
            PointerSlots, Stream)
            .string();

    std::function<void(const CapturedGlobals &,
                       llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &,
                       void **,
                       typename DeviceTraits<VendorTypes>::DeviceStream_t)>
        CaptureEpilogue =
            [this, DynamicHash, StaticHash, MnemeDir](
                const CapturedGlobals &GlobalVars,
                llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>>
                    &DeviceMemory,
                void **Args,
                typename DeviceTraits<VendorTypes>::DeviceStream_t Stream) {
              std::filesystem::path Filename(
                  MnemeDir / (std::string("DeviceState.epilogue.") +
                              std::to_string(StaticHash) + "." +
                              std::to_string(DynamicHash) + ".mneme"));

              Instances[DynamicHash].EpilogueFn =
                  MnemeSnapshot<VendorTypes>::takeMnemeSnapshot(
                      GlobalVars, DeviceMemory, Filename, KernelArgSizes, Args,
                      PointerSlots, Stream)
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

public:
  RecordDatabase() : KernelWhiteList(""), HasRegex(false) {
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
  std::optional<std::function<
      void(const CapturedGlobals &,
           llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &, void **,
           typename DeviceTraits<VendorTypes>::DeviceStream_t)>>
  takeSnapshot(
      void *VAddr, uint64_t VASize, proteus::JITKernelInfo &KInfo,
      const CapturedGlobals &GlobalVars, ::JitDeviceImplT &Proteus,
      llvm::ArrayRef<KernelArgPointerSlot> PointerSlots,
      llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &DeviceMemory,
      dim3 &GridDim, dim3 &BlockDim, void **Args, size_t SharedMem,
      typename DeviceTraits<VendorTypes>::DeviceStream_t Stream) {
    using namespace proteus;

    if (!shouldRecord(KInfo.getName())) {
      LOG_INFO("Skip record of Kernel");
      return std::nullopt;
    }

    auto IT = KernelRecords.try_emplace(
        KInfo.getStaticHash().getValue(),
        KernelInstancesCollection(getDir(), VAddr, VASize, KInfo, PointerSlots,
                                  MaxRecordings));
    LOG_INFO("Created instance");
    IT.first->second.addGlobalSidecarModules(MnemeDirectory,
                                             KInfo.getStaticHash().getValue(),
                                             KInfo, Proteus, GlobalVars);
    IT.first->second.addCapturedGlobalNames(GlobalVars);
    return IT.first->second.takeSnapshot<VendorTypes>(
        MnemeDirectory, GlobalVars, DeviceMemory, GridDim, BlockDim, Args,
        SharedMem, Stream, KInfo.getStaticHash().getValue());
  }

  const std::string getDir() const { return MnemeDirectory.string(); }
};

} // namespace mneme
