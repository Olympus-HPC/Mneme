#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeDeviceKernels.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeMemory.hpp"
#include "mneme/MnemePageManager.hpp"
#include "mneme/MnemeSnapshot.hpp"
#include "mneme/MnemeUtils.hpp"

namespace mneme {

enum class ReplayResetMode { Bytes, Diff };

inline ReplayResetMode parseReplayResetMode(std::string Mode) {
  std::transform(Mode.begin(), Mode.end(), Mode.begin(),
                 [](unsigned char C) { return std::tolower(C); });
  if (Mode == "bytes" || Mode == "full")
    return ReplayResetMode::Bytes;
  if (Mode == "diff")
    return ReplayResetMode::Diff;
  LOG_FATAL("Unknown Mneme replay reset mode: " + Mode +
            ". Expected 'bytes' or 'diff'.");
  return ReplayResetMode::Bytes;
}

inline size_t getEnvSizeOrDefault(const char *Name, size_t Default) {
  const char *Value = std::getenv(Name);
  if (!Value || !*Value)
    return Default;
  try {
    return std::stoull(Value);
  } catch (...) {
    LOG_WARN("Invalid value '{}' for {}, using default {}", Value, Name,
             Default);
    return Default;
  }
}

template <DeviceVendors VendorTypes> class PrologueState;
template <DeviceVendors VendorTypes> class EpilogueState;

// Abstract base for a replay memory state. A concrete state is either a
// prologue (kernel input state) or an epilogue (expected kernel output state).
// Subclasses provide load(), which materializes the state onto the device.
template <DeviceVendors VendorTypes> class ReplayMemState {
public:
  using MnemeDeviceRT = DeviceTraits<VendorTypes>;
  using DeviceError_t = typename MnemeDeviceRT::DeviceError_t;
  using DeviceStream_t = typename MnemeDeviceRT::DeviceStream_t;
  using KernelFunction_t = typename MnemeDeviceRT::KernelFunction_t;
  using DeviceModule_t = typename DeviceTraits<VendorTypes>::DeviceModule_t;

protected:
  std::shared_ptr<KernelInfo> KInfo;
  llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> DeviceMemoryState;
  std::unordered_map<std::string, ReplayGlobalVar> GlobalVars;
  std::unique_ptr<void *[]> Args;

  explicit ReplayMemState(Snapshot<VendorTypes> SnapshotState)
      : KInfo(std::move(SnapshotState.KInfo)),
        DeviceMemoryState(std::move(SnapshotState.DeviceMemory)),
        GlobalVars(std::move(SnapshotState.GlobalVars)) {
    LOG_DEBUG("Initialized replay memory state for kernel {}",
              KInfo->getName());
    Args = copyOutArgs();
  }

  void copyToDevice() {
    for (auto &[DevAddr, MemBlob] : DeviceMemoryState) {
      LOG_DEBUG("Copying {} from Address {} to device address {} {}",
                isPrologue() ? "Prologue" : "Epilogue",
                (void *)MemBlob.getHostData().get(), MemBlob.getBlobAddr(),
                MemBlob.getSize());
      auto CEC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceCopy(
          MemBlob.getBlobAddr(), MemBlob.getHostData().get(), MemBlob.getSize(),
          MnemeDeviceRT::MemcpyHostToDeviceKind()));
      if (CEC)
        LOG_FATAL("Could not copy Memory Blob to device EC: " + CEC.value() +
                  "\n");
    }
  }

  void copyGlobals() {
    for (auto &[GVName, GVI] : GlobalVars) {
      LOG_DEBUG("Copying data of variable {} to device addr {} and of size {}",
                GVName, GVI.DevAddr, GVI.VarSize);
      auto CEC = MnemeDeviceRT::DeviceErrorCheck(
          MnemeDeviceRT::DeviceCopy(GVI.DevAddr, GVI.HostAddr, GVI.VarSize,
                                    MnemeDeviceRT::MemcpyHostToDeviceKind()));
      if (CEC)
        LOG_FATAL("Could not copy global " + GVName +
                  " to device EC: " + CEC.value() + "\n");
    }
  }

  // Distinguishes the two concrete roles for diagnostic logging.
  virtual bool isPrologue() const = 0;

private:
  std::unique_ptr<void *[]> copyOutArgs() const {
    void **Args = new void *[KInfo->getNumArgs()];
    auto ArgData = KInfo->getArgData();
    for (int I = 0; I < getNumArgs(); I++) {
      Args[I] = ArgData[I].get();
    }
    std::unique_ptr<void *[]> ArgUniquePtr{Args};
    return ArgUniquePtr;
  }

public:
  ReplayMemState() = delete;
  virtual ~ReplayMemState() { release(); }

  virtual void load() = 0;

  void reset() {
    copyToDevice();
    copyGlobals();
  }

  void release() {
    for (auto &[DevAddr, MemBlob] : DeviceMemoryState) {
      auto EC = DeviceTraits<VendorTypes>::DeviceErrorCheck(MemBlob.release());
      if (EC)
        LOG_WARN("Could not release replay memory blob: {}", EC.value());
    }
    DeviceMemoryState.clear();
  }

  const llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> &
  getDeviceMemory() const {
    return DeviceMemoryState;
  }

  const std::unordered_map<std::string, ReplayGlobalVar> &getGlobalVars() const {
    return GlobalVars;
  }

  // RTTI-free downcasts to a concrete role.
  virtual PrologueState<VendorTypes> *asPrologue() { return nullptr; }
  virtual EpilogueState<VendorTypes> *asEpilogue() { return nullptr; }

  void **getArgs() const { return reinterpret_cast<void **>(Args.get()); }

  uint64_t getNumArgs() const { return KInfo->getNumArgs(); }

  void initializeGlobals(DeviceModule_t VendorMod) {
    LOG_INFO("Initializing {} Globals", GlobalVars.size());
    for (auto &KV : GlobalVars) {
      auto [LoadedAddr, LoadedSize] =
          DeviceTraits<VendorTypes>::getGlobalAddrFromModule(VendorMod,
                                                             KV.first);
      if (KV.second.DevAddr != LoadedAddr) {
        LOG_WARN("Global : {} was loaded on different addresses Record:{} vs "
                 "Replay:{}",
                 KV.first, KV.second.DevAddr, LoadedAddr);
        KV.second.DevAddr = LoadedAddr;
      }

      if (KV.second.VarSize != LoadedSize)
        LOG_FATAL("Global :" + KV.first +
                  "has a different size between record and replay\n" +
                  "Record Size:" + std::to_string(KV.second.VarSize) +
                  "\nReplay Size:" + std::to_string(LoadedSize));

      auto EC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
          DeviceTraits<VendorTypes>::DeviceCopy(
              KV.second.DevAddr, KV.second.HostAddr, KV.second.VarSize,
              DeviceTraits<VendorTypes>::MemcpyHostToDeviceKind()));
      if (EC)
        LOG_FATAL("Copying Global :" + KV.first +
                  " from host to device raised error\nEC: " + EC.value());
      LOG_INFO("Successfully loaded global variable: {}", KV.first);
    }
  }
};

// Replay state for the recorded kernel input. Besides the full byte reset
// inherited from the base, a prologue can carry a diff reset plan: the byte
// ranges the kernel mutated (taken from a diff epilogue snapshot), coalesced,
// packed into device buffers, and restored in-place by a device scatter
// kernel between replay iterations.
template <DeviceVendors VendorTypes>
class PrologueState : public ReplayMemState<VendorTypes> {
public:
  using MnemeDeviceRT = DeviceTraits<VendorTypes>;
  using DeviceStream_t = typename MnemeDeviceRT::DeviceStream_t;

  PrologueState(const std::string &KernelName, const std::string &SnapshotFile)
      : ReplayMemState<VendorTypes>(
            MnemeSnapshot<VendorTypes>::readBytesSnapshot(KernelName,
                                                          SnapshotFile)) {}

  ~PrologueState() override { releaseScatterPlan(); }

  void load() override {
    for (auto &[DevAddr, MemBlob] : this->DeviceMemoryState) {
      auto EC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
          MemBlob.map(DevAddr, MemBlob.getActualSize(), MemBlob.getSize()));
      if (EC)
        LOG_FATAL("Error raised during mapping prologue memeory:" + EC.value());

      if (DevAddr != reinterpret_cast<void *>(MemBlob.getBlobAddr()))
        LOG_FATAL("Could not map Record Address " +
                  util::pointerToHexString(DevAddr) +
                  " instead ReplayInstance got " +
                  util::pointerToHexString(
                      static_cast<uint8_t *>(MemBlob.getBlobAddr())) +
                  "\n");
    }

    this->copyToDevice();
  }

  PrologueState<VendorTypes> *asPrologue() override { return this; }

  // Builds (or clears) the diff reset plan for this prologue from the
  // epilogue's diff snapshot. With ReplayResetMode::Bytes the plan is simply
  // released and reset() falls back to the full byte copy.
  void prepareResetPlan(const EpilogueState<VendorTypes> &Epilogue,
                        ReplayResetMode Mode) {
    buildDiffResetPlan(Epilogue, Mode);
  }

  using ReplayMemState<VendorTypes>::reset;

  void reset(ReplayResetMode Mode, DeviceStream_t Stream = 0) {
    if (Mode == ReplayResetMode::Diff) {
      resetFromDiffPlan(Stream);
      return;
    }
    reset();
  }

protected:
  bool isPrologue() const override { return true; }

private:
  struct ResetSpan {
    void *Dst = nullptr;
    const uint8_t *Src = nullptr;
    size_t Size = 0;
  };

  std::vector<ResetSpan> DiffResetSpans;
  std::unique_ptr<uint8_t[]> HostScatterData;
  DiffResetScatterTask *DeviceScatterTasks = nullptr;
  uint8_t *DeviceScatterData = nullptr;
  size_t ScatterTaskCount = 0;
  size_t ScatterDataSize = 0;

  void releaseScatterPlan() {
    if (DeviceScatterTasks) {
      auto EC = MnemeDeviceRT::DeviceErrorCheck(
          MnemeDeviceRT::DeviceFree(DeviceScatterTasks));
      if (EC)
        LOG_FATAL("Could not release Mneme diff scatter task buffer EC: " +
                  EC.value());
      DeviceScatterTasks = nullptr;
    }
    if (DeviceScatterData) {
      auto EC = MnemeDeviceRT::DeviceErrorCheck(
          MnemeDeviceRT::DeviceFree(DeviceScatterData));
      if (EC)
        LOG_FATAL("Could not release Mneme diff scatter data buffer EC: " +
                  EC.value());
      DeviceScatterData = nullptr;
    }
    HostScatterData.reset();
    ScatterTaskCount = 0;
    ScatterDataSize = 0;
  }

  static std::vector<typename MnemeSnapshot<VendorTypes>::DiffRange>
  coalesceRanges(
      std::vector<typename MnemeSnapshot<VendorTypes>::DiffRange> Ranges,
      size_t MaxGap) {
    if (Ranges.empty())
      return Ranges;
    std::sort(Ranges.begin(), Ranges.end(),
              [](auto &LHS, auto &RHS) { return LHS.Offset < RHS.Offset; });

    std::vector<typename MnemeSnapshot<VendorTypes>::DiffRange> Coalesced;
    auto Current = Ranges.front();
    for (size_t I = 1; I < Ranges.size(); ++I) {
      auto &Next = Ranges[I];
      size_t CurrentEnd = Current.Offset + Current.Size;
      if (Next.Offset <= CurrentEnd + MaxGap) {
        size_t NextEnd = Next.Offset + Next.Size;
        Current.Size = std::max(CurrentEnd, NextEnd) - Current.Offset;
      } else {
        Coalesced.push_back(Current);
        Current = Next;
      }
    }
    Coalesced.push_back(Current);
    return Coalesced;
  }

  void addScatterSpans(
      const std::vector<typename MnemeSnapshot<VendorTypes>::DiffRange> &Ranges,
      const uint8_t *HostBase, uint8_t *DeviceBase, size_t MaxGap) {
    auto Spans = coalesceRanges(Ranges, MaxGap);
    for (auto &Range : Spans) {
      DiffResetSpans.push_back(ResetSpan{DeviceBase + Range.Offset,
                                         HostBase + Range.Offset, Range.Size});
    }
  }

  void prepareDeviceScatterPlan(size_t TaskBytes) {
    releaseScatterPlan();
    if (DiffResetSpans.empty())
      return;
    if (TaskBytes == 0)
      TaskBytes = 4096;

    std::vector<DiffResetScatterTask> HostTasks;
    HostTasks.reserve(DiffResetSpans.size());
    for (auto &Span : DiffResetSpans)
      ScatterDataSize += Span.Size;
    HostScatterData = std::make_unique<uint8_t[]>(ScatterDataSize);

    size_t PackedOffset = 0;
    for (auto &Span : DiffResetSpans) {
      size_t Offset = 0;
      while (Offset < Span.Size) {
        size_t ChunkSize = std::min(TaskBytes, Span.Size - Offset);
        std::memcpy(HostScatterData.get() + PackedOffset, Span.Src + Offset,
                    ChunkSize);
        HostTasks.push_back(DiffResetScatterTask{
            static_cast<uint8_t *>(Span.Dst) + Offset,
            reinterpret_cast<const uint8_t *>(PackedOffset), ChunkSize});
        PackedOffset += ChunkSize;
        Offset += ChunkSize;
      }
    }

    if (PackedOffset != ScatterDataSize)
      LOG_FATAL("Internal Mneme diff scatter packing size mismatch");

    auto EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceMalloc(
        reinterpret_cast<void **>(&DeviceScatterData), ScatterDataSize));
    if (EC)
      LOG_FATAL("Could not allocate Mneme diff scatter data buffer EC: " +
                EC.value());

    EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceCopy(
        DeviceScatterData, HostScatterData.get(), ScatterDataSize,
        MnemeDeviceRT::MemcpyHostToDeviceKind()));
    if (EC)
      LOG_FATAL("Could not copy Mneme diff scatter data buffer EC: " +
                EC.value());
    HostScatterData.reset();

    for (auto &Task : HostTasks)
      Task.Src = DeviceScatterData + reinterpret_cast<uintptr_t>(Task.Src);

    ScatterTaskCount = HostTasks.size();
    EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceMalloc(
        reinterpret_cast<void **>(&DeviceScatterTasks),
        ScatterTaskCount * sizeof(DiffResetScatterTask)));
    if (EC)
      LOG_FATAL("Could not allocate Mneme diff scatter task buffer EC: " +
                EC.value());

    EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceCopy(
        DeviceScatterTasks, HostTasks.data(),
        ScatterTaskCount * sizeof(DiffResetScatterTask),
        MnemeDeviceRT::MemcpyHostToDeviceKind()));
    if (EC)
      LOG_FATAL("Could not copy Mneme diff scatter task buffer EC: " +
                EC.value());
  }

  bool buildDiffResetPlan(const EpilogueState<VendorTypes> &Epilogue,
                          ReplayResetMode Mode) {
    releaseScatterPlan();
    DiffResetSpans.clear();

    if (Mode == ReplayResetMode::Bytes)
      return false;

    const std::string &EpilogueSnapshot = Epilogue.getSnapshotPath();
    std::string Error;
    auto DiffPlan =
        MnemeSnapshot<VendorTypes>::readDiffPlan(EpilogueSnapshot, &Error);
    if (!DiffPlan) {
      LOG_FATAL("Could not use diff reset for " + EpilogueSnapshot + ": " +
                (Error.empty() ? "epilogue is not a diff snapshot" : Error));
      return false;
    }

    if (DiffPlan->Globals.size() != this->GlobalVars.size()) {
      Error = "diff global count does not match prologue";
    } else if (DiffPlan->Blobs.size() != this->DeviceMemoryState.size()) {
      Error = "diff blob count does not match prologue";
    }

    if (!Error.empty()) {
      LOG_FATAL("Could not use diff reset: " + Error);
      return false;
    }

    size_t MaxGap =
        getEnvSizeOrDefault("MNEME_REPLAY_DIFF_SCATTER_MAX_GAP_BYTES", 0);
    size_t ScatterTaskBytes =
        getEnvSizeOrDefault("MNEME_REPLAY_DIFF_SCATTER_TASK_BYTES", 4096);

    for (auto &Global : DiffPlan->Globals) {
      auto It = this->GlobalVars.find(Global.Name);
      if (It == this->GlobalVars.end()) {
        Error = "diff references global missing from prologue: " + Global.Name;
        break;
      }
      auto &GVI = It->second;
      if (GVI.VarSize != Global.VarSize) {
        Error = "diff global size mismatch for: " + Global.Name;
        break;
      }
      addScatterSpans(Global.Ranges, static_cast<const uint8_t *>(GVI.HostAddr),
                      static_cast<uint8_t *>(GVI.DevAddr), MaxGap);
    }

    if (Error.empty()) {
      for (auto &BlobPlan : DiffPlan->Blobs) {
        auto It = this->DeviceMemoryState.find(BlobPlan.DevAddr);
        if (It == this->DeviceMemoryState.end()) {
          Error = "diff references device allocation missing from prologue";
          break;
        }
        auto &Blob = It->second;
        if (Blob.getActualSize() != BlobPlan.ActualSize ||
            Blob.getSize() != BlobPlan.Size) {
          Error = "diff memory blob size mismatch";
          break;
        }
        addScatterSpans(BlobPlan.Ranges, Blob.getHostData().get(),
                        static_cast<uint8_t *>(Blob.getBlobAddr()), MaxGap);
      }
    }

    if (!Error.empty()) {
      LOG_FATAL("Could not use diff reset: " + Error);
      return false;
    }

    prepareDeviceScatterPlan(ScatterTaskBytes);

    LOG_INFO("Prepared Mneme diff reset plan");
    return true;
  }

  void resetFromDiffPlan(DeviceStream_t Stream) {
    if (ScatterTaskCount == 0)
      return;
    auto EC = MnemeDeviceRT::DeviceErrorCheck(
        launchDiffResetScatterKernel<VendorTypes>(DeviceScatterTasks,
                                                  ScatterTaskCount, Stream));
    if (EC)
      LOG_FATAL("Could not launch Mneme diff scatter reset kernel EC: " +
                EC.value());
    EC = MnemeDeviceRT::DeviceErrorCheck(
        MnemeDeviceRT::DeviceStreamSynchronize(Stream));
    if (EC)
      LOG_FATAL("Could not synchronize Mneme diff scatter reset kernel EC: " +
                EC.value());
  }
};

// Replay state for the expected kernel output. Unlike the prologue, load()
// allocates fresh device memory rather than mapping to recorded addresses.
template <DeviceVendors VendorTypes>
class EpilogueState : public ReplayMemState<VendorTypes> {
public:
  explicit EpilogueState(Snapshot<VendorTypes> SnapshotState,
                         std::string SnapshotPath = "")
      : ReplayMemState<VendorTypes>(std::move(SnapshotState)),
        SnapshotPath(std::move(SnapshotPath)) {}

  void load() override {
    for (auto &[DevAddr, MemBlob] : this->DeviceMemoryState) {
      auto EC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
          MemBlob.allocate(MemBlob.getSize()));
      if (EC)
        LOG_FATAL("Error raised during mapping prologue memeory:" + EC.value());
    }
    this->copyToDevice();
  }

  EpilogueState<VendorTypes> *asEpilogue() override { return this; }

  // Path of the snapshot file this state was reconstructed from. A diff reset
  // plan re-reads this file to learn which byte ranges the kernel mutated.
  const std::string &getSnapshotPath() const { return SnapshotPath; }

  // Verifies a replayed prologue against this expected-output epilogue. At call
  // time the prologue's device buffers hold the kernel's actual output.
  virtual bool matches(const PrologueState<VendorTypes> &Prologue) const {
    LOG_DEBUG("Comparing memory states");
    bool Correct = true;

    // Device memory blobs: both states are device-resident, so the blob
    // comparator reads both device addresses directly.
    for (auto &[DevAddr, ProBlob] : Prologue.getDeviceMemory()) {
      auto It = this->DeviceMemoryState.find(DevAddr);
      if (It == this->DeviceMemoryState.end()) {
        LOG_WARN("Cannot find {} in comparators", DevAddr);
        return false;
      }
      auto &EpiBlob = It->second;
      if (EpiBlob.getSize() != ProBlob.getSize()) {
        LOG_WARN("Sizes Differ {} vs {}", ProBlob.getSize(), EpiBlob.getSize());
        return false;
      }
      if (EpiBlob != ProBlob)
        Correct = false;
    }

    // Global variables: only the prologue's globals are device-resident; the
    // epilogue never loads globals.
    for (auto &[GVName, ProGV] : Prologue.getGlobalVars()) {
      auto It = this->GlobalVars.find(GVName);
      if (It == this->GlobalVars.end()) {
        LOG_WARN("comparing with global var {} that exists only on one of the "
                 "comparators",
                 GVName);
        Correct = false;
        continue;
      }

      auto &EpiGV = It->second;
      std::unique_ptr<uint8_t[]> ProData(new uint8_t[ProGV.VarSize]);
      auto CEC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
          DeviceTraits<VendorTypes>::DeviceCopy(
              ProData.get(), ProGV.DevAddr, ProGV.VarSize,
              DeviceTraits<VendorTypes>::MemcpyDeviceToHostKind()));
      if (CEC)
        LOG_FATAL("Could not copy global from device EC: " + CEC.value() +
                  "\n");

      if (memcmp(EpiGV.HostAddr, ProData.get(), ProGV.VarSize) != 0)
        Correct = false;
    }

    LOG_DEBUG("Memory States {}", Correct ? "are the same" : "differ");
    return Correct;
  }

protected:
  bool isPrologue() const override { return false; }

private:
  std::string SnapshotPath;
};

template <DeviceVendors VendorTypes>
std::unique_ptr<ReplayMemState<VendorTypes>>
makeReplayPrologueState(const std::string &KernelName,
                        const std::string &SnapshotFile) {
  return std::make_unique<PrologueState<VendorTypes>>(KernelName, SnapshotFile);
}

template <DeviceVendors VendorTypes>
std::unique_ptr<ReplayMemState<VendorTypes>>
makeReplayEpilogueState(const std::string &KernelName,
                        const std::string &SnapshotFile,
                        const std::string &BasePrologueFile) {
  Snapshot<VendorTypes> Snap =
      MnemeSnapshot<VendorTypes>::openSnapshot(SnapshotFile)
          ->reconstruct(KernelName, BasePrologueFile);
  return std::make_unique<EpilogueState<VendorTypes>>(std::move(Snap),
                                                      SnapshotFile);
}

} // namespace mneme
