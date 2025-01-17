#include "DeviceTraits.hpp"
#include "MnemeSnapshot.hpp"
#include "MnemeSymbols.hpp"
#include "Utils.hpp"
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>

namespace mneme {
template <typename MemBlobT, DeviceVendors VendorTypes> class ReplayMemState {
public:
  using DeviceError_t = typename DeviceTraits<VendorTypes>::DeviceError_t;
  using DeviceStream_t = typename DeviceTraits<VendorTypes>::DeviceStream_t;
  using KernelFunction_t = typename DeviceTraits<VendorTypes>::KernelFunction_t;

  enum InstanceType { Prologue, Epilogue };

  KernelInfo KInfo;
  llvm::DenseMap<void *, MemBlobT> DeviceMemoryState;
  llvm::DenseMap<std::string, GlobalVarInfo> GlobalVars;
  InstanceType IType;
  std::string SnapshotName;

private:
  void loadPrologueMemory() {
    for (auto &[DevAddr, MemBlob] : DeviceMemoryState) {
      auto EC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
          MemBlob.allocate(DevAddr, MemBlob.Size));
      if (EC)
        FATAL_ERROR("Error raised during mapping prologue memeory:" + EC.get());

      if (DevAddr != MemBlob.BlobAddr)
        FATAL_ERROR("Could not map Record Address " +
                    util::pointerToHexString(DevAddr) +
                    " instead ReplayInstance got " +
                    util::pointerToHexString(MemBlob.BlobAddr) + "\n");

      // Copy data to device
      auto CEC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
          MemBlobT::DeviceCopy(DevAddr, MemBlob.HostAddr, MemBlob.Size,
                               MemBlobT::MemcpyHostToDeviceKind()));
      if (CEC)
        FATAL_ERROR("Could not copy Memory Blob to device EC: " + CEC.get() +
                    "\n");
    }
  }

  void loadEpilogueMemory() {
    // TODO: This is currently empty, and we will perform verification on host.
    // if the host verification ends up being a bottleneck, we should copy to
    // device but do not provide a map address. So, the data will be allocated
    // into some random memory location (not the same as the prologue). Once we
    // have this we can compare in GPU.
  }

public:
  ReplayMemState(std::string KernelName, std::string SnapshotName,
                 InstanceType IType)
      : KInfo(nullptr, KernelName), SnapshotName(SnapshotName), IType(IType) {
    MnemeSnapshot<MemBlobT, VendorTypes>::readMnemeSnapShot(
        SnapshotName, GlobalVars, DeviceMemoryState, &KInfo);
    DBG(Logger::logs("mneme")
        << "Initialized Snapshot of kernel " << KInfo.getName() << "of state "
        << (IType == InstanceType::Prologue ? "Prologue" : "Epilogue") << "\n");
  }

  void load(llvm::DenseMap<std::string, void *> LoadedGlobals) {
    // On replay, we first need to verify that the module's loaded globals
    // reside at the same address as the ones at replay time.
    for (auto &[GlobalName, GlobalDeviceAddress] : LoadedGlobals) {
      auto it = GlobalVars.find(GlobalName);
      if (it == GlobalVars.end())
        FATAL_ERROR("Cannot find global " + GlobalName +
                    " in persistent snapshot");
      auto &RecordedGlobal = it->second;
      if (RecordedGlobal.DevAddr != GlobalDeviceAddress)
        Logger::warn() << " Replay Global Address (" << std::hex
                       << GlobalDeviceAddress << std::dec
                       << ") differs than recorded address (" << std::hex
                       << RecordedGlobal.DevAddr
                       << ") ... \n This can result in incorrect execution. "
                          "Check verification\n";

      // Globals are not loaded in the case of Epilogue instances.
      // We will verify the values of the globals, after kernel execution.
      // But there is no case that we cannot reach the epilogue state
      if (IType == InstanceType::Epilogue)
        continue;

      auto Res =
          DeviceTraits<VendorTypes>::DeviceErrorCheck(MemBlobT::DeviceCopy(
              GlobalDeviceAddress, RecordedGlobal.HostAddr,
              RecordedGlobal.VarSize, MemBlobT::MemcpyHostToDeviceKind()));
      if (Res)
        FATAL_ERROR("Could not copy Global " + GlobalName + " to device ");
    }

    if (IType == InstanceType::Prologue)
      loadPrologueMemory();
    else
      loadEpilogueMemory();
  }
};

template <typename MemBlobT, DeviceVendors VendorTypes>
class ReplayInstance : public mneme::KernelInstance {
  std::string KernelName;
  std::string DemangledName;
  ReplayMemState<MemBlobT, VendorTypes> PrologueState;
  ReplayMemState<MemBlobT, VendorTypes> EpilogueState;
  llvm::SmallVector<std::string> ModuleFiles;

private:
  static dim3 getDim3(llvm::json::Object &Info, std::string key) {
    auto JObject = Info.getObject(key);
    if (!JObject)
      FATAL_ERROR("Could not load " + key + " from dimensions");

    long x = *JObject->getInteger("x");
    long y = *JObject->getInteger("y");
    long z = *JObject->getInteger("z");

    return dim3(x, y, z);
  }

public:
  ReplayInstance(std::string JSONDbFn, std::string InstanceID) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> KernelInfoMB =
        llvm::MemoryBuffer::getFile(JSONDbFn, /* isText */ true,
                                    /* RequiresNullTerminator */ true);
    if (!KernelInfoMB)
      FATAL_ERROR("Error occurred: " + KernelInfoMB.getError().message() +
                  "\n");

    llvm::json::Value JsonInfo =
        llvm::cantFail(llvm::json::parse(KernelInfoMB.get()->getBuffer()),
                       "Cannot convert buffer to json value");
    auto *JSONRoot = JsonInfo.getAsObject();

    auto extractStringValue = [&](llvm::json::Object *Obj, std::string Key) {
      auto JValue = Obj->getString(Key);
      if (!JValue)
        FATAL_ERROR("Could not read '" + Key + "' from json file");
      return *JValue;
    };

    KernelName = extractStringValue(JSONRoot, "KernelName");
    DemangledName = extractStringValue(JSONRoot, "DemangledName");

    llvm::json::Array *RecordedModules = JSONRoot->getArray("Modules");
    for (auto Mod : *RecordedModules) {
      ModuleFiles.emplace_back(Mod.getAsString());
    }

    auto InstanceJSON = JSONRoot->getObject(InstanceID);
    if (!InstanceJSON)
      FATAL_ERROR("Cannot load instance " + InstanceID);
    BlockDim = getDim3(*InstanceJSON, "BlockDims");
    GridDim = getDim3(*InstanceJSON, "GridDims");
    PrologueFn = extractStringValue(InstanceJSON, "Prologue");
    EpilogueFn = extractStringValue(InstanceJSON, "Epilogue");
  }
};

} // namespace mneme
