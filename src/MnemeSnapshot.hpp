#pragma once
#include "DeviceTraits.hpp"
#include "MnemeMemory.hpp"
#include "MnemeSymbols.hpp"
#include "Utils.hpp"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <filesystem>
#include <functional>
#include <hip/hip_runtime.h>
#include <llvm/ADT/SmallVector.h>
#include <memory>
#include <optional>

#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/ADT/StableHashing.h>
#include <llvm/ADT/StringRef.h>
#include <string>
#include <sys/types.h>
namespace mneme {

template <typename ImplT, typename MemBlobT, DeviceVendors VendorTypes>
class MnemeSnapshot {
  using DeviceError_t = typename DeviceTraits<VendorTypes>::DeviceError_t;
  using DeviceStream_t = typename DeviceTraits<VendorTypes>::DeviceStream_t;
  using KernelFunction_t = typename DeviceTraits<VendorTypes>::KernelFunction_t;

public:
  std::filesystem::path static takeMnemeSnapshot(
      llvm::SmallVector<GlobalVarInfo> &GlobalVars,
      llvm::DenseMap<void *, MemBlobT> &DeviceMemory,
      std::filesystem::path &MnemeDir, uint64_t DynamicHash,
      std::shared_ptr<KernelInfo> KInfo, void **Args, size_t SharedMem,
      DeviceStream_t Stream, bool IsPrologue = true) {
    DBG(Logger::logs("mneme") << "KInfo is " << KInfo.get() << "\n");
    llvm::stable_hash KHash = llvm::stable_hash_combine_string(KInfo->Name);
    std::error_code EC;
    std::filesystem::path Filename(
        MnemeDir /
        (std::string("DeviceState") +
         std::string(IsPrologue ? ".prologue." : ".epilogue.") +
         std::to_string(KHash) + "." + std::to_string(DynamicHash) + ".mneme"));
    // Syncrhonize cause we need to get a consistent GPU state.
    // We may want to do a DeviceSynchronize().
    auto DEC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
        ImplT::DeviceStreamSynchronize(Stream));
    if (DEC)
      FATAL_ERROR("Synnchronizing stream  failed");
    llvm::raw_fd_ostream OutBC(Filename.string(), EC);
    // First write Global Variables.
    for (auto &GV : GlobalVars) {
      GV.HostAddr = static_cast<void *>(new uint8_t[GV.VarSize]);
      auto DEC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
          MemBlobT::DeviceCopy(GV.HostAddr, GV.DevAddr, GV.VarSize,
                               MemBlobT::MemcpyHostToDeviceKind()));
      if (DEC)
        FATAL_ERROR(
            "Copying from device to host for global variables failed\n");
      OutBC << GV;

      delete[] static_cast<uint8_t *>(GV.HostAddr);
      GV.HostAddr = nullptr;
    }
    // Write the Device Memory
    for (auto &[Ptr, Blob] : DeviceMemory)
      OutBC << Blob;
    // Lastly write the arguments
    size_t NumArgs = KInfo->KernelArgs.size();
    OutBC << llvm::StringRef(reinterpret_cast<const char *>(&NumArgs),
                             sizeof(NumArgs));
    for (int I = 0; I < NumArgs; I++) {
      OutBC << llvm::StringRef(
          reinterpret_cast<const char *>(&KInfo->KernelArgs[I]),
          sizeof(size_t));
      OutBC << llvm::StringRef(reinterpret_cast<const char *>(Args[I]),
                               KInfo->KernelArgs[I]);
    }

    return std::filesystem::canonical(Filename);
  }
};

struct KernelInstance {
  std::string PrologueFn;
  std::string EpilogueFn;
  dim3 BlockDim;
  dim3 GridDim;
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
    return instance;
  }
  KernelInstance(dim3 &GridDim, dim3 &BlockDim)
      : GridDim(GridDim), BlockDim(BlockDim) {}
  KernelInstance() = default;
};

class KernelInstancesCollection {
  std::shared_ptr<KernelInfo> KInfo;
  llvm::DenseMap<uint64_t, KernelInstance> Instances;

public:
  llvm::json::Object toJSON() const {
    llvm::json::Object Collection;
    Collection["KernelName"] = KInfo->getName();
    Collection["DemangedName"] = llvm::demangle(KInfo->getName());
    Collection["Modules"] = llvm::json::Array(KInfo->ModuleFiles);
    llvm::json::Object JSONInstances;
    for (auto &[hash, KI] : Instances) {
      JSONInstances[std::to_string(hash)] = KI.toJSON();
    }
    Collection["instances"] = std::move(JSONInstances);
    return Collection;
  }

  KernelInstancesCollection(std::shared_ptr<KernelInfo> KInfo) : KInfo(KInfo) {}

  llvm::stable_hash computeHash(dim3 &GridDim, dim3 &BlockDim) {
    auto BlockHash = llvm::stable_hash_combine((llvm::stable_hash)BlockDim.x,
                                               (llvm::stable_hash)BlockDim.y,
                                               (llvm::stable_hash)BlockDim.z);
    auto GridHash = llvm::stable_hash_combine((llvm::stable_hash)GridDim.x,
                                              (llvm::stable_hash)GridDim.y,
                                              (llvm::stable_hash)GridDim.z);
    return llvm::stable_hash_combine(GridHash, BlockHash);
  }

  bool shouldSnapshot(dim3 &GridDim, dim3 &BlockDim) {
    return !Instances.contains(computeHash(GridDim, BlockDim));
  }

  template <typename ImplT, typename MemBlobT, DeviceVendors VendorTypes>
  std::optional<std::function<void(
      llvm::SmallVector<GlobalVarInfo> &, llvm::DenseMap<void *, MemBlobT> &,
      void **, size_t, typename DeviceTraits<VendorTypes>::DeviceStream_t)>>
  takeSnapshot(std::filesystem::path &MnemeDir,
               llvm::SmallVector<GlobalVarInfo> &GlobalVars,
               llvm::DenseMap<void *, MemBlobT> &DeviceMemory, dim3 &GridDim,
               dim3 &BlockDim, void **Args, size_t SharedMem,
               typename DeviceTraits<VendorTypes>::DeviceStream_t Stream) {
    auto hash = computeHash(GridDim, BlockDim);

    if (Instances.contains(hash))
      return std::nullopt;

    DBG(Logger ::logs("mneme") << "Capturing prologue for Kernel with hash: "
                               << std::hex << this << std::dec << "\n");
    Instances.insert({hash, KernelInstance(GridDim, BlockDim)});
    Instances[hash].PrologueFn =
        MnemeSnapshot<ImplT, MemBlobT, VendorTypes>::takeMnemeSnapshot(
            GlobalVars, DeviceMemory, MnemeDir, hash, KInfo, Args, SharedMem,
            Stream)
            .string();

    std::function<void(llvm::SmallVector<GlobalVarInfo> &,
                       llvm::DenseMap<void *, MemBlobT> &, void **, size_t,
                       typename DeviceTraits<VendorTypes>::DeviceStream_t)>
        CaptureEpilogue = [this, hash, &MnemeDir](
                              llvm::SmallVector<GlobalVarInfo> &GlobalVars,
                              llvm::DenseMap<void *, MemBlobT> &DeviceMemory,
                              void **Args, size_t SharedMem,
                              typename DeviceTraits<VendorTypes>::DeviceStream_t
                                  Stream) {
          Instances[hash].EpilogueFn =
              MnemeSnapshot<ImplT, MemBlobT, VendorTypes>::takeMnemeSnapshot(
                  GlobalVars, DeviceMemory, MnemeDir, hash, KInfo, Args,
                  SharedMem, Stream, true)
                  .string();
        };
    return CaptureEpilogue;
  }
};

class RecordDatabase {
  std::filesystem::path MnemeDirectory;
  llvm::DenseMap<uint64_t, bool> IsRecordedKernel;
  llvm::DenseMap<uint64_t, KernelInstancesCollection> KernelRecords;

public:
  RecordDatabase() {
    auto Dir = std::getenv("RR_DATA_DIR");
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
      auto JSONRecord = Record.toJSON();
      llvm::raw_fd_ostream JsonOS(JsonFilename.string(), EC);
      JsonOS << llvm::json::Value(std::move(JSONRecord));
      JsonOS.close();
    }
  }

  bool shouldRecord(KernelInfo &KI) {
    // We always return true here.
    if (!IsRecordedKernel.contains(KI.StaticHash))
      return IsRecordedKernel[KI.StaticHash];

    return true;
  }

  template <typename ImplT, typename MemBlobT, DeviceVendors VendorTypes>
  auto takeSnapshot(std::shared_ptr<KernelInfo> KInfo,
                    llvm::SmallVector<GlobalVarInfo> &GlobalVars,
                    llvm::DenseMap<void *, MemBlobT> &DeviceMemory,
                    dim3 &GridDim, dim3 &BlockDim, void **Args,
                    size_t SharedMem,
                    typename DeviceTraits<VendorTypes>::DeviceStream_t Stream) {
    auto IT = KernelRecords.try_emplace(KInfo->StaticHash,
                                        KernelInstancesCollection(KInfo));
    return IT.first->second.takeSnapshot<ImplT, MemBlobT, VendorTypes>(
        MnemeDirectory, GlobalVars, DeviceMemory, GridDim, BlockDim, Args,
        SharedMem, Stream);
  }

  std::string getDir() const { return MnemeDirectory.string(); }
};

} // namespace mneme
