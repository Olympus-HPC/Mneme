#include <cstdint>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <memory>
#include <string>

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeMemory.hpp"
#include "mneme/MnemeSnapshot.hpp"
#include "mneme/MnemeUtils.hpp"

namespace mneme {

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
  llvm::DenseMap<uint64_t, MnemeMemoryBlob<VendorTypes>> DeviceMemoryState;
  std::unordered_map<std::string, ReplayGlobalVar> GlobalVars;
  std::unique_ptr<void *[]> Args;
  uintptr_t ReplayVABase = 0;

  explicit ReplayMemState(Snapshot<VendorTypes> SnapshotState,
                          uintptr_t ReplayVABase)
      : KInfo(std::move(SnapshotState.KInfo)),
        DeviceMemoryState(std::move(SnapshotState.DeviceMemory)),
        GlobalVars(std::move(SnapshotState.GlobalVars)),
        ReplayVABase(ReplayVABase) {
    LOG_DEBUG("Initialized replay memory state for kernel {}",
              KInfo->getName());
    Args = copyOutArgs();
  }

  void copyToDevice() {
    for (auto &[BlobId, MemBlob] : DeviceMemoryState) {
      LOG_DEBUG("Copying {} blob id {} offset {} from host address {} to device address {} size {}",
                isPrologue() ? "Prologue" : "Epilogue",
                BlobId, MemBlob.getBlobOffset(),
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

  MnemeMemoryBlob<VendorTypes> *findBlobById(uint64_t BlobId) {
    auto It = DeviceMemoryState.find(BlobId);
    if (It == DeviceMemoryState.end())
      return nullptr;
    return &It->second;
  }

  void materializeManagedPointerArgs() {
    for (size_t I = 0; I < KInfo->ArgData.size(); ++I) {
      if (KInfo->getArgEncodingKinds()[I] !=
          KernelArgEncodingKind::ManagedPointer)
        continue;

      auto *Blob = findBlobById(KInfo->getManagedArgBlobId(I));
      if (!Blob)
        LOG_FATAL("Managed pointer kernel arg references unknown blob id");

      auto Offset = KInfo->getManagedArgOffset(I);
      if (Offset >= Blob->getSize())
        LOG_FATAL("Managed pointer kernel arg offset exceeds blob size");

      auto *ReplayPtr =
          static_cast<uint8_t *>(Blob->getBlobAddr()) + Offset;
      KInfo->materializeManagedPointerArg(I, ReplayPtr);
    }
  }

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
    for (auto &[BlobId, MemBlob] : DeviceMemoryState) {
      auto EC = DeviceTraits<VendorTypes>::DeviceErrorCheck(MemBlob.release());
      if (EC)
        LOG_WARN("Could not release replay memory blob id {} offset {}: {}",
                 BlobId, MemBlob.getBlobOffset(), EC.value());
    }
    DeviceMemoryState.clear();
  }

  const llvm::DenseMap<uint64_t, MnemeMemoryBlob<VendorTypes>> &
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

// Replay state for the recorded kernel input.
template <DeviceVendors VendorTypes>
class PrologueState : public ReplayMemState<VendorTypes> {
public:
  PrologueState(const std::string &KernelName, const std::string &SnapshotFile,
                uintptr_t ReplayVABase)
      : ReplayMemState<VendorTypes>(
            MnemeSnapshot<VendorTypes>::readBytesSnapshot(KernelName,
                                                          SnapshotFile),
            ReplayVABase) {}

  void load() override {
    for (auto &[BlobId, MemBlob] : this->DeviceMemoryState) {
      auto *ReplayAddr = reinterpret_cast<void *>(this->ReplayVABase +
                                                  MemBlob.getBlobOffset());
      LOG_DEBUG("Mapping prologue blob id {} at replay VA {} using offset {}",
                BlobId, ReplayAddr, MemBlob.getBlobOffset());
      auto EC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
          MemBlob.map(ReplayAddr, MemBlob.getActualSize(), MemBlob.getSize()));
      if (EC)
        LOG_FATAL("Error raised during mapping prologue memeory:" + EC.value());
    }

    this->materializeManagedPointerArgs();
    this->copyToDevice();
  }

  PrologueState<VendorTypes> *asPrologue() override { return this; }

protected:
  bool isPrologue() const override { return true; }
};

// Replay state for the expected kernel output. Unlike the prologue, load()
// allocates fresh device memory rather than mapping to recorded addresses.
template <DeviceVendors VendorTypes>
class EpilogueState : public ReplayMemState<VendorTypes> {
public:
  explicit EpilogueState(Snapshot<VendorTypes> SnapshotState,
                         uintptr_t ReplayVABase)
      : ReplayMemState<VendorTypes>(std::move(SnapshotState), ReplayVABase) {}

  void load() override {
    for (auto &[BlobId, MemBlob] : this->DeviceMemoryState) {
      auto EC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
          MemBlob.allocate(MemBlob.getSize()));
      if (EC)
        LOG_FATAL("Error raised during mapping prologue memeory:" + EC.value());
    }
    this->copyToDevice();
  }

  EpilogueState<VendorTypes> *asEpilogue() override { return this; }

  // Verifies a replayed prologue against this expected-output epilogue. At call
  // time the prologue's device buffers hold the kernel's actual output.
  virtual bool matches(const PrologueState<VendorTypes> &Prologue) const {
    LOG_DEBUG("Comparing memory states");
    bool Correct = true;

    // Device memory blobs: both states are device-resident, so the blob
    // comparator reads both device addresses directly.
    for (auto &[BlobId, ProBlob] : Prologue.getDeviceMemory()) {
      auto It = this->DeviceMemoryState.find(BlobId);
      if (It == this->DeviceMemoryState.end()) {
        LOG_WARN("Cannot find blob id {} in comparators", BlobId);
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
};

template <DeviceVendors VendorTypes>
std::unique_ptr<ReplayMemState<VendorTypes>>
makeReplayPrologueState(const std::string &KernelName,
                        const std::string &SnapshotFile,
                        uintptr_t ReplayVABase) {
  return std::make_unique<PrologueState<VendorTypes>>(
      KernelName, SnapshotFile, ReplayVABase);
}

template <DeviceVendors VendorTypes>
std::unique_ptr<ReplayMemState<VendorTypes>>
makeReplayEpilogueState(const std::string &KernelName,
                        const std::string &SnapshotFile,
                        const std::string &BasePrologueFile,
                        uintptr_t ReplayVABase) {
  Snapshot<VendorTypes> Snap =
      MnemeSnapshot<VendorTypes>::openSnapshot(SnapshotFile)
          ->reconstruct(KernelName, BasePrologueFile);
  return std::make_unique<EpilogueState<VendorTypes>>(
      std::move(Snap), ReplayVABase);
}

} // namespace mneme
