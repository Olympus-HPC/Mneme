#pragma once
#include <cstdint>
#include <filesystem>
#include <functional>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <optional>
#include <regex>

#include "llvm/Demangle/Demangle.h"
#include <llvm/ADT/StableHashing.h>
#include <llvm/ADT/StringRef.h>
#include <string>
#include <sys/types.h>

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeKernelInfo.hpp"
#include "mneme/MnemeLLVMUtils.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeMemory.hpp"
#include "mneme/MnemeSymbols.hpp"
#include "mneme/MnemeUtils.hpp"

namespace mneme {

template <DeviceVendors VendorTypes> class MnemeSnapshot {
  using MnemeDeviceRT = DeviceTraits<VendorTypes>;
  using DeviceError_t = typename MnemeDeviceRT::DeviceError_t;
  using DeviceStream_t = typename MnemeDeviceRT::DeviceStream_t;
  using KernelFunction_t = typename MnemeDeviceRT::KernelFunction_t;

public:
  std::filesystem::path static takeMnemeSnapshot(
      llvm::SmallVector<GlobalVarInfo> &GlobalVars,
      llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &DeviceMemory,
      std::filesystem::path &Filename, std::shared_ptr<KernelInfo> KInfo,
      void **Args, DeviceStream_t Stream) {
    LOG_DEBUG("Storing mneme snapshot: {}", Filename.string());
    llvm::stable_hash KHash = llvm::stable_hash_combine_string(KInfo->Name);
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
    LOG_DEBUG("Number of Globals in snapshot:{} stored at position:{}",
              TotalGlobals, OutBC.tell());

    OutBC << llvm::StringRef(reinterpret_cast<const char *>(&TotalGlobals),
                             sizeof(size_t));
    for (auto &GV : GlobalVars) {
      auto DEC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
          DeviceTraits<VendorTypes>::DeviceCopy(
              GV.HostAddr.get(), GV.DevAddr, GV.VarSize,
              DeviceTraits<VendorTypes>::MemcpyDeviceToHostKind()));
      if (DEC)
        LOG_FATAL("Copying from device to host for global variables failed\n");
      OutBC << GV;
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
    size_t NumArgs = KInfo->KernelArgSizes.size();
    LOG_DEBUG("Number of Kernel Arguments in snapshot:{} stored at position:{}",
              NumArgs, OutBC.tell());

    OutBC << llvm::StringRef(reinterpret_cast<const char *>(&NumArgs),
                             sizeof(NumArgs));

    for (int I = 0; I < NumArgs; I++) {
      OutBC << llvm::StringRef(
          reinterpret_cast<const char *>(&KInfo->KernelArgSizes[I]),
          sizeof(size_t));
      OutBC << llvm::StringRef(reinterpret_cast<const char *>(Args[I]),
                               KInfo->KernelArgSizes[I]);
    }

    return std::filesystem::canonical(Filename);
  }

  void static readMnemeSnapShot(
      std::string Filename,
      llvm::DenseMap<std::string, GlobalVarInfo> &GlobalVars,
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
    auto *CurrentPtr = Start;
    size_t TotalGlobals = util::extractScalar<size_t>(CurrentPtr);
    LOG_DEBUG("Snapshot contains {} Globals at location {}", TotalGlobals,
              (uintptr_t)CurrentPtr - (uintptr_t)Start);
    for (auto I = 0; I < TotalGlobals; I++) {
      auto GV = GlobalVarInfo::fromBuffer(CurrentPtr);
      GlobalVars.insert({GV.Name, std::move(GV)});
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
  KernelInstance(std::shared_ptr<KernelInfo> KInfo, dim3 &GridDim,
                 dim3 &BlockDim, uint64_t SharedMem, void **Args)
      : GridDim(GridDim), BlockDim(BlockDim), SharedMem(SharedMem),
        NumOccurrences(0) {
    auto CanSpecialize = KInfo->getArgSpecializations();
    auto toDouble = KInfo->getToDoubleFunc();
    for (int I = 0; I < KInfo->getNumArgs(); I++) {
      if (CanSpecialize[I]) {
        ArgValues.emplace_back(toDouble[I](Args[I]));
      } else {
        ArgValues.emplace_back(-1.0);
      }
    }
  }
  KernelInstance() = default;
};

class KernelInstancesCollection {
  std::shared_ptr<KernelInfo> KInfo;
  void *VAddr;
  uint64_t VASize;
  llvm::DenseMap<uint64_t, KernelInstance> Instances;

public:
  llvm::json::Object toJSON(uint64_t StaticHash) const {
    llvm::json::Object Collection;
    Collection["StaticHash"] = StaticHash;
    Collection["VAddr"] =
        util::pointerToHexString(reinterpret_cast<uint8_t *>(VAddr));
    Collection["VASize"] = VASize;
    auto &KName = KInfo->getName();
    Collection["KernelName"] = KName;
    std::size_t pos = KName.find("__intern__");
    std::string Orig =
        (pos != std::string::npos) ? KName.substr(0, pos) : KName;
    auto &Executable = KInfo->getExecutable();
    Collection["DemangledName"] = llvm::demangle(Orig);
    Collection["Modules"] = llvm::json::Array(Executable.getModuleIRFiles());
    Collection["BinaryBlobs"] =
        llvm::json::Array(Executable.getModuleBinFiles());
    Collection["ArgNames"] = llvm::json::Array(KInfo->getArgNames());
    Collection["Specializations"] =
        llvm::json::Array(KInfo->getArgSpecializations());
    llvm::json::Object JSONInstances;
    for (auto &[hash, KI] : Instances) {
      JSONInstances[std::to_string(hash)] = KI.toJSON();
    }
    Collection["instances"] = std::move(JSONInstances);
    return Collection;
  }

  KernelInstancesCollection(void *VAddr, uint64_t VASize,
                            std::shared_ptr<KernelInfo> KInfo)
      : VAddr(VAddr), VASize(VASize), KInfo(KInfo) {
    // Here I need to extract information regarding the kernel
    auto Func = KInfo->getExecutable().getKernelFunction(KInfo->Name);
    if (!Func)
      LOG_FATAL("Cannot find descriptor of function {}", KInfo->Name);

    KInfo->setArgSizes(mneme::getFuncDescr(*Func.value()));
    KInfo->setArgNames(mneme::getArgNames(*Func.value()));
    KInfo->setSpecializations(mneme::canSpecialize(*Func.value()));
    KInfo->setToDoubleFunc(mneme::convertToDouble(*Func.value()));
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

    auto CanSpecialize = KInfo->getArgSpecializations();
    auto ArgSizes = KInfo->getArgSizes();
    for (int I = 0; I < KInfo->getNumArgs(); I++) {
      if (CanSpecialize[I]) {
        DHash = llvm::stable_hash_combine(
            DHash, stable_hash_combine_string(
                       llvm::StringRef((char *)Args[I], ArgSizes[I])));
      }
    }
    return DHash;
  }

  template <DeviceVendors VendorTypes>
  std::optional<std::function<
      void(llvm::SmallVector<GlobalVarInfo> &,
           llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &, void **,
           typename DeviceTraits<VendorTypes>::DeviceStream_t)>>
  takeSnapshot(
      std::filesystem::path &MnemeDir,
      llvm::SmallVector<GlobalVarInfo> &GlobalVars,
      llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &DeviceMemory,
      dim3 &GridDim, dim3 &BlockDim, void **Args, size_t SharedMem,
      typename DeviceTraits<VendorTypes>::DeviceStream_t Stream,
      uint64_t StaticHash) {
    auto DynamicHash = computeHash(GridDim, BlockDim, SharedMem, Args);

    if (Instances.contains(DynamicHash)) {
      Instances[DynamicHash].NumOccurrences++;
      LOG_DEBUG(
          "Kernel {} with DynamicHash {} is already recorded, skipping ...",
          KInfo->getName(), DynamicHash);
      return std::nullopt;
    }

    LOG_DEBUG("First Instance of Kernel {} with DynamicHash {}, recording ...",
              KInfo->getName(), DynamicHash);

    Instances.insert({DynamicHash, KernelInstance(KInfo, GridDim, BlockDim,
                                                  SharedMem, Args)});
    std::filesystem::path Filename(MnemeDir /
                                   (std::string("DeviceState.prologue.") +
                                    std::to_string(StaticHash) + "." +
                                    std::to_string(DynamicHash) + ".mneme"));

    Instances[DynamicHash].PrologueFn =
        MnemeSnapshot<VendorTypes>::takeMnemeSnapshot(
            GlobalVars, DeviceMemory, Filename, KInfo, Args, Stream)
            .string();

    std::function<void(llvm::SmallVector<GlobalVarInfo> &,
                       llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &,
                       void **,
                       typename DeviceTraits<VendorTypes>::DeviceStream_t)>
        CaptureEpilogue =
            [this, DynamicHash, StaticHash, &MnemeDir](
                llvm::SmallVector<GlobalVarInfo> &GlobalVars,
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
                      GlobalVars, DeviceMemory, Filename, KInfo, Args, Stream)
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

public:
  RecordDatabase() : KernelWhiteList(""), HasRegex(false) {
    auto Dir = std::getenv("MNEME_DATA_DIR");
    auto WhiteList = std::getenv("MNEME_RR_KERNELS");
    if (WhiteList) {
      HasRegex = true;
      RegexStr = std::string(WhiteList);
      KernelWhiteList = std::string(WhiteList);
    }

    MnemeDirectory =
        (Dir ? std::string(Dir) : std::filesystem::current_path().string());

    if (!std::filesystem::is_directory(MnemeDirectory)) {
      throw std::runtime_error("Path :" + MnemeDirectory.string() +
                               " does not exist.\n");
    }
    MnemeDirectory = std::filesystem::absolute(MnemeDirectory);
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
      return std::regex_search(KernelName, KernelWhiteList);
    } catch (const std::regex_error &e) {
      LOG_WARN("Invalid regex: {}, ... falling back and recording everything");
    }
    return true;
  }

  template <DeviceVendors VendorTypes>
  std::optional<std::function<
      void(llvm::SmallVector<GlobalVarInfo> &,
           llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &, void **,
           typename DeviceTraits<VendorTypes>::DeviceStream_t)>>
  takeSnapshot(
      void *VAddr, uint64_t VASize, std::shared_ptr<KernelInfo> KInfo,
      llvm::SmallVector<GlobalVarInfo> &GlobalVars,
      llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &DeviceMemory,
      dim3 &GridDim, dim3 &BlockDim, void **Args, size_t SharedMem,
      typename DeviceTraits<VendorTypes>::DeviceStream_t Stream) {

    if (!shouldRecord(KInfo->getName())) {
      LOG_INFO("Skip record of Kernel {}, not matching regex",
               KInfo->getName());
      return std::nullopt;
    }

    auto IT = KernelRecords.try_emplace(
        KInfo->getStaticHash(),
        KernelInstancesCollection(VAddr, VASize, KInfo));
    return IT.first->second.takeSnapshot<VendorTypes>(
        MnemeDirectory, GlobalVars, DeviceMemory, GridDim, BlockDim, Args,
        SharedMem, Stream, KInfo->getStaticHash());
  }

  std::string getDir() const { return MnemeDirectory.string(); }
};

} // namespace mneme
