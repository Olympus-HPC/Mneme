#include "DeviceTraits.hpp"
#include "MnemeSnapshot.hpp"
#include "MnemeSymbols.hpp"
#include "Utils.hpp"
#include <algorithm>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <memory>
#include <proteus/Utils.h>

namespace mneme {
template <typename MemBlobT, DeviceVendors VendorTypes> class ReplayMemState {
public:
  using MnemeDeviceRT = DeviceTraits<VendorTypes>;
  using DeviceError_t = typename MnemeDeviceRT::DeviceError_t;
  using DeviceStream_t = typename MnemeDeviceRT::DeviceStream_t;
  using KernelFunction_t = typename MnemeDeviceRT::KernelFunction_t;

  enum InstanceType { Prologue, Epilogue };

  std::shared_ptr<KernelInfo> KInfo;
  llvm::DenseMap<void *, MemBlobT> DeviceMemoryState;
  llvm::DenseMap<std::string, GlobalVarInfo> GlobalVars;
  InstanceType IType;
  std::string SnapshotName;

private:
  void loadPrologueMemory() {
    for (auto &[DevAddr, MemBlob] : DeviceMemoryState) {
      auto EC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
          MemBlob.allocate(DevAddr, MemBlob.getSize()));
      if (EC)
        FATAL_ERROR("Error raised during mapping prologue memeory:" +
                    EC.value());

      if (DevAddr != reinterpret_cast<void *>(MemBlob.getBlobAddr()))
        FATAL_ERROR("Could not map Record Address " +
                    util::pointerToHexString(DevAddr) +
                    " instead ReplayInstance got " +
                    util::pointerToHexString(
                        static_cast<uint8_t *>(MemBlob.getBlobAddr())) +
                    "\n");

      // Copy data to device
      auto CEC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceCopy(
          DevAddr, MemBlob.getHostData().get(), MemBlob.getSize(),
          MnemeDeviceRT::MemcpyHostToDeviceKind()));
      if (CEC)
        FATAL_ERROR("Could not copy Memory Blob to device EC: " + CEC.value() +
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
  ReplayMemState() = default;
  ReplayMemState(std::string KernelName, std::string SnapshotName,
                 InstanceType IType)
      : KInfo(std::make_shared<KernelInfo>(nullptr, KernelName)),
        SnapshotName(SnapshotName), IType(IType) {
    MnemeSnapshot<VendorTypes>::readMnemeSnapShot(SnapshotName, GlobalVars,
                                                  DeviceMemoryState, KInfo);
    DBG(Logger::logs("mneme")
        << "Initialized Snapshot of kernel " << KInfo->getName() << "of state "
        << (IType == InstanceType::Prologue ? "Prologue" : "Epilogue") << "\n");
  }

  void load() {
    if (IType == InstanceType::Prologue)
      loadPrologueMemory();
    else
      loadEpilogueMemory();
  }
};

template <typename MemBlobT, DeviceVendors VendorTypes>
class ReplayInstance : public mneme::KernelInstance {
  using DeviceMemState = ReplayMemState<MemBlobT, VendorTypes>;
  using DeviceModule_t = typename DeviceTraits<VendorTypes>::DeviceModule_t;
  std::string KernelName;
  std::string DemangledName;
  DeviceMemState PrologueState;
  DeviceMemState EpilogueState;
  llvm::SmallVector<std::string> ModuleFileNames;

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
      auto Module = Mod.getAsString();
      if (!Module)
        FATAL_ERROR("Could not read Module value");

      ModuleFileNames.emplace_back(Module.value());
    }

    auto Instances = JSONRoot->getObject("instances");
    if (!Instances)
      FATAL_ERROR("Corrupted JSON file, does not contain instances entry\n");

    auto InstanceJSON = Instances->getObject(InstanceID);
    if (!InstanceJSON)
      FATAL_ERROR("Cannot load instance " + InstanceID);
    BlockDim = getDim3(*InstanceJSON, "BlockDims");
    GridDim = getDim3(*InstanceJSON, "GridDims");
    PrologueFn = extractStringValue(InstanceJSON, "Prologue");
    EpilogueFn = extractStringValue(InstanceJSON, "Epilogue");

    PrologueState = DeviceMemState(KernelName, PrologueFn,
                                   DeviceMemState::InstanceType::Prologue);
    EpilogueState = DeviceMemState(KernelName, EpilogueFn,
                                   DeviceMemState::InstanceType::Epilogue);
  }

  llvm::ArrayRef<std::string> getModules() const { return ModuleFileNames; }
  std::string &getKernelName() { return KernelName; }
  llvm::SmallVector<std::unique_ptr<llvm::Module>>
  loadModules(llvm::LLVMContext &Ctx) {
    llvm::SmallVector<std::unique_ptr<llvm::Module>> RecordedModules;
    for (auto &Fn : ModuleFileNames) {
      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Buffer =
          llvm::MemoryBuffer::getFile(Fn);
      if (!Buffer)
        FATAL_ERROR("Error with loading file " + Fn +
                    "\n Error Code:" + Buffer.getError().message());

      llvm::Expected<std::unique_ptr<llvm::Module>> ModuleOrErr =
          llvm::parseBitcodeFile(Buffer->get()->getMemBufferRef(), Ctx);

      if (!ModuleOrErr)
        FATAL_ERROR("Error parsing bitcode: " +
                    llvm::toString(ModuleOrErr.takeError()));

      RecordedModules.emplace_back(std::move(ModuleOrErr.get()));
    }
    return RecordedModules;
  }

  dim3 getRecordedGrid() const { return GridDim; }
  dim3 getRecordedBlock() const { return BlockDim; }

  void initializeGlobals(DeviceModule_t VendorMod) {
    for (auto &KV : PrologueState.GlobalVars) {
      auto [LoadedAddr, LoadedSize] =
          DeviceTraits<VendorTypes>::getGlobalAddrFromModule(VendorMod,
                                                             KV.first);
      if (KV.second.DevAddr != LoadedAddr) {
        FATAL_ERROR(
            "Global :" + KV.first +
            " was loaded on different address between record and replay\n" +
            "Record Address:" + util::pointerToHexString(KV.second.DevAddr) +
            "\n" + "Replay Address:" + util::pointerToHexString(LoadedAddr));
      }

      if (KV.second.VarSize != LoadedSize)
        FATAL_ERROR("Global :" + KV.first +
                    "has a different size between record and replay\n" +
                    "Record Size:" + std::to_string(KV.second.VarSize) +
                    "\nReplay Size:" + std::to_string(LoadedSize));

      auto EC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
          DeviceTraits<VendorTypes>::DeviceCopy(
              KV.second.DevAddr, KV.second.HostAddr.get(), KV.second.VarSize,
              DeviceTraits<VendorTypes>::MemcpyHostToDeviceKind()));
      if (EC)
        FATAL_ERROR("Copying Global :" + KV.first +
                    " from host to device raised error\nEC: " + EC.value());
      DBG(Logger::logs("mneme")
          << "Successfully loaded global variable " << KV.first << "\n");
    }
  }

  void initializeDeviceMemory() { PrologueState.load(); }
};

} // namespace mneme
