#include <cstdint>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <memory>

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeMemory.hpp"
#include "mneme/MnemePageManager.hpp"
#include "mneme/MnemeSnapshot.hpp"
#include "mneme/MnemeUtils.hpp"

namespace mneme {
template <DeviceVendors VendorTypes> class ReplayMemState {
public:
  using MnemeDeviceRT = DeviceTraits<VendorTypes>;
  using DeviceError_t = typename MnemeDeviceRT::DeviceError_t;
  using DeviceStream_t = typename MnemeDeviceRT::DeviceStream_t;
  using KernelFunction_t = typename MnemeDeviceRT::KernelFunction_t;
  using DeviceModule_t = typename DeviceTraits<VendorTypes>::DeviceModule_t;

  enum InstanceType { Prologue, Epilogue };

  std::shared_ptr<KernelInfo> KInfo;
  llvm::DenseMap<void *, MnemeMemoryBlob<VendorTypes>> DeviceMemoryState;
  std::unordered_map<std::string, ReplayGlobalVar> GlobalVars;
  InstanceType IType;
  std::string SnapshotName;
  std::unique_ptr<void *[]> Args;

private:
  void copyToDevice() {
    for (auto &[DevAddr, MemBlob] : DeviceMemoryState) {
      LOG_DEBUG("Copying {} from Address {} to device address {} {}",
                IType == InstanceType::Prologue ? "Prologue" : "Epilogue",
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

  void loadPrologueMemory() {
    for (auto &[DevAddr, MemBlob] : DeviceMemoryState) {
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

    copyToDevice();
  }

  void loadEpilogueMemory() {
    for (auto &[DevAddr, MemBlob] : DeviceMemoryState) {
      auto EC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
          MemBlob.allocate(MemBlob.getSize()));
      if (EC)
        LOG_FATAL("Error raised during mapping prologue memeory:" + EC.value());
    }
    copyToDevice();
  }

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
  ReplayMemState() = default;
  ReplayMemState(std::string KernelName, std::string SnapshotName,
                 InstanceType IType)
      : KInfo(std::make_shared<KernelInfo>(KernelName)),
        SnapshotName(SnapshotName), IType(IType) {
    MnemeSnapshot<VendorTypes>::readMnemeSnapShot(SnapshotName, GlobalVars,
                                                  DeviceMemoryState, KInfo);
    LOG_DEBUG("Initialized Snapshot for kernel {} of state {}",
              KInfo->getName(),
              (IType == InstanceType::Prologue ? "Prologue" : "Epilogue"));
    Args = copyOutArgs();
  }

  void load() {
    if (IType == InstanceType::Prologue)
      loadPrologueMemory();
    else
      loadEpilogueMemory();
  }

  void reset() {
    copyToDevice();
    copyGlobals();
  }

  void release() {
    for (auto &[DevAddr, MemBlob] : DeviceMemoryState) {
      MemBlob.release();
    }
    DeviceMemoryState.clear();
  }

  // Overload equality operator
  bool operator==(const ReplayMemState<VendorTypes> &other) const {
    LOG_DEBUG("Comparing memory states");
    auto &OtherBlob = other.DeviceMemoryState;
    bool correct = true;
    for (auto &[DevAddr, MemBlob] : DeviceMemoryState) {
      auto it = OtherBlob.find(DevAddr);
      if (it == OtherBlob.end()) {
        LOG_WARN("Cannot find {} in comparators", DevAddr);
        return false;
      }
      if (MemBlob.getSize() != it->second.getSize()) {
        LOG_WARN("Sizes Differ {} vs {}", MemBlob.getSize(),
                 it->second.getSize());
        return false;
      }

      if (!DeviceTraits<VendorTypes>::compareDeviceBlobs(
              (const char *)it->second.getBlobAddr(),
              (const char *)MemBlob.getBlobAddr(), MemBlob.getSize()))
        correct = false;
    }

    for (auto &[GVName, GVI] : GlobalVars) {
      auto it = other.GlobalVars.find(GVName);
      if (it == other.GlobalVars.end()) {
        LOG_WARN("comparing with global var {} that exists only on one of the "
                 "comparators",
                 GVName);
        correct = false;
        continue;
      }

      auto &OtherGV = it->second;

      std::unique_ptr<uint8_t[]> hostData(new uint8_t[GVI.VarSize]);
      void *comparator;
      if (IType == InstanceType::Prologue) {
        auto CEC = MnemeDeviceRT::DeviceErrorCheck(
            MnemeDeviceRT::DeviceCopy(hostData.get(), GVI.DevAddr, GVI.VarSize,
                                      MnemeDeviceRT::MemcpyDeviceToHostKind()));
        if (CEC)
          LOG_FATAL("Could not copy Memory Blob to device EC: " + CEC.value() +
                    "\n");
        comparator = OtherGV.HostAddr;
      } else if (other.IType == InstanceType::Prologue) {
        auto CEC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceCopy(
            OtherGV.HostAddr, OtherGV.DevAddr, OtherGV.VarSize,
            MnemeDeviceRT::MemcpyDeviceToHostKind()));
        if (CEC)
          LOG_FATAL("Could not copy Memory Blob to device EC: " + CEC.value() +
                    "\n");
        comparator = GVI.HostAddr;
      } else {
        LOG_FATAL("Either this or other need to be of Prologue type");
      }

      if (memcmp(comparator, hostData.get(), GVI.VarSize) != 0)
        correct = false;
    }
    LOG_DEBUG("Memory States {}", correct ? "are the same" : "differ");

    return correct;
  }
  // Derive inequality operator
  bool operator!=(const ReplayMemState<VendorTypes> &other) const {
    return !(*this == other);
  }

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

} // namespace mneme
