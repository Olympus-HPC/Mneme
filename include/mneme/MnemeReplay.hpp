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
#include "mneme/MnemePageManager.hpp"
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
  std::unique_ptr<Snapshot<VendorTypes>> Snap;
  std::unique_ptr<void *[]> Args;
  // Start of the VA reservation at record time and at replay time.
  uintptr_t RecordedVABase;
  uintptr_t ReplayVABase;

  explicit ReplayMemState(std::unique_ptr<Snapshot<VendorTypes>> Snap,
                          uintptr_t RecordedVABase, uintptr_t ReplayVABase)
      : Snap(std::move(Snap)), RecordedVABase(RecordedVABase),
        ReplayVABase(ReplayVABase) {
    LOG_DEBUG("Initialized replay memory state for kernel {}",
              this->Snap->KInfo->getName());
    Args = copyOutArgs();
  }

  void copyToDevice() {
    for (auto &[BlobId, MemBlob] : Snap->DeviceMemory) {
      LOG_DEBUG("Copying {} blob id {} from host address {} to device "
                "address {} size {}",
                isPrologue() ? "Prologue" : "Epilogue", BlobId,
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
    for (auto &[GVName, GVI] : Snap->GlobalVars) {
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

  // Fills in pointer arguments once every blob has a replay address.
  void materializeManagedPointerArgs() {
    auto &KInfo = *Snap->KInfo;
    auto Kinds = KInfo.getArgEncodingKinds();
    for (size_t I = 0; I < Kinds.size(); ++I) {
      if (Kinds[I] != KernelArgEncodingKind::ManagedPointer)
        continue;

      auto BlobId = KInfo.getManagedArgBlobId(I);
      auto It = Snap->DeviceMemory.find(BlobId);
      if (It == Snap->DeviceMemory.end())
        LOG_FATAL("Kernel arg " + std::to_string(I) +
                  " references unknown blob id " + std::to_string(BlobId));

      auto &Blob = It->second;
      auto Offset = KInfo.getManagedArgOffset(I);
      if (Offset >= Blob.getSize())
        LOG_FATAL("Kernel arg " + std::to_string(I) + " offset " +
                  std::to_string(Offset) + " exceeds blob id " +
                  std::to_string(BlobId) + " size");

      KInfo.materializeManagedPointerArg(
          I, static_cast<uint8_t *>(Blob.getBlobAddr()) + Offset);
    }
  }

private:
  std::unique_ptr<void *[]> copyOutArgs() const {
    void **Args = new void *[Snap->KInfo->getNumArgs()];
    auto ArgData = Snap->KInfo->getArgData();
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
    for (auto &[BlobId, MemBlob] : Snap->DeviceMemory) {
      auto EC = DeviceTraits<VendorTypes>::DeviceErrorCheck(MemBlob.release());
      if (EC)
        LOG_WARN("Could not release replay memory blob id {}: {}", BlobId,
                 EC.value());
    }
    Snap->DeviceMemory.clear();
  }

  const llvm::DenseMap<uint64_t, MnemeMemoryBlob<VendorTypes>> &
  getDeviceMemory() const {
    return Snap->DeviceMemory;
  }

  const std::unordered_map<std::string, ReplayGlobalVar> &getGlobalVars() const {
    return Snap->GlobalVars;
  }

  // RTTI-free downcasts to a concrete role.
  virtual PrologueState<VendorTypes> *asPrologue() { return nullptr; }
  virtual EpilogueState<VendorTypes> *asEpilogue() { return nullptr; }

  void **getArgs() const { return reinterpret_cast<void **>(Args.get()); }

  uint64_t getNumArgs() const { return Snap->KInfo->getNumArgs(); }

  void initializeGlobals(DeviceModule_t VendorMod) {
    LOG_INFO("Initializing {} Globals", Snap->GlobalVars.size());
    for (auto &KV : Snap->GlobalVars) {
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
                uintptr_t RecordedVABase, uintptr_t ReplayVABase)
      : ReplayMemState<VendorTypes>(
            BaseSnapshotSource<VendorTypes>(SnapshotFile).load(KernelName),
            RecordedVABase, ReplayVABase) {}

  void load() override {
    auto &Snap = *this->Snap;
    Snap.checkReplayVABase(this->RecordedVABase, this->ReplayVABase);

    for (auto &[BlobId, MemBlob] : Snap.DeviceMemory) {
      auto *ReplayAddr = Snap.replayBlobAddress(MemBlob, this->ReplayVABase);
      LOG_DEBUG("Mapping prologue blob id {} at replay address {}", BlobId,
                ReplayAddr);
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
  EpilogueState(std::unique_ptr<Snapshot<VendorTypes>> Snap,
                uintptr_t RecordedVABase, uintptr_t ReplayVABase)
      : ReplayMemState<VendorTypes>(std::move(Snap), RecordedVABase,
                                    ReplayVABase) {}

  void load() override {
    for (auto &[BlobId, MemBlob] : this->Snap->DeviceMemory) {
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
      auto It = this->Snap->DeviceMemory.find(BlobId);
      if (It == this->Snap->DeviceMemory.end()) {
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
      auto It = this->Snap->GlobalVars.find(GVName);
      if (It == this->Snap->GlobalVars.end()) {
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
                        uintptr_t RecordedVABase, uintptr_t ReplayVABase) {
  return std::make_unique<PrologueState<VendorTypes>>(
      KernelName, SnapshotFile, RecordedVABase, ReplayVABase);
}

template <DeviceVendors VendorTypes>
std::unique_ptr<ReplayMemState<VendorTypes>>
makeReplayEpilogueState(const std::string &KernelName,
                        const std::string &SnapshotFile,
                        const std::string &BasePrologueFile,
                        uintptr_t RecordedVABase, uintptr_t ReplayVABase) {
  auto Snap =
      SnapshotFormatRegistry<VendorTypes>::open(SnapshotFile)
          ->read(KernelName, BaseSnapshotSource<VendorTypes>(BasePrologueFile));
  return std::make_unique<EpilogueState<VendorTypes>>(
      std::move(Snap), RecordedVABase, ReplayVABase);
}

} // namespace mneme
