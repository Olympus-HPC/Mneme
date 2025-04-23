#include "MnemeLogger.hpp"
#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeMemory.hpp"
#include "mneme/MnemePageManager.hpp"
#include "mneme/MnemeSnapshot.hpp"
#include "mneme/MnemeSymbols.hpp"
#include "mneme/Utils.hpp"
#include <cstdint>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <memory>
#include <proteus/Utils.h>

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
  llvm::DenseMap<std::string, GlobalVarInfo> GlobalVars;
  InstanceType IType;
  std::string SnapshotName;

private:
  void copyToDevice() {
    for (auto &[DevAddr, MemBlob] : DeviceMemoryState) {
      // Copy data to device
      auto CEC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceCopy(
          MemBlob.getBlobAddr(), MemBlob.getHostData().get(), MemBlob.getSize(),
          MnemeDeviceRT::MemcpyHostToDeviceKind()));
      if (CEC)
        FATAL_ERROR("Could not copy Memory Blob to device EC: " + CEC.value() +
                    "\n");
    }
  }

  void copyGlobals() {
    for (auto &[GVName, GVI] : GlobalVars) {
      LOG_DEBUG("Copying data of variable {} to device addr {} and of size {}",
                GVName, GVI.DevAddr, GVI.VarSize);
      auto CEC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceCopy(
          GVI.DevAddr, GVI.HostAddr.get(), GVI.VarSize,
          MnemeDeviceRT::MemcpyHostToDeviceKind()));
      if (CEC)
        FATAL_ERROR("Could not copy global " + GVI.Name +
                    " to device EC: " + CEC.value() + "\n");
    }
  }

  void loadPrologueMemory() {
    for (auto &[DevAddr, MemBlob] : DeviceMemoryState) {
      auto EC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
          MemBlob.map(DevAddr, MemBlob.getActualSize(), MemBlob.getSize()));
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
    }

    copyToDevice();
  }

  void loadEpilogueMemory() {
    for (auto &[DevAddr, MemBlob] : DeviceMemoryState) {
      auto EC = DeviceTraits<VendorTypes>::DeviceErrorCheck(
          MemBlob.allocate(MemBlob.getSize()));
      if (EC)
        FATAL_ERROR("Error raised during mapping prologue memeory:" +
                    EC.value());
    }
    copyToDevice();
  }

public:
  ReplayMemState() = default;
  ReplayMemState(std::string KernelName, std::string SnapshotName,
                 InstanceType IType)
      : KInfo(std::make_shared<KernelInfo>(nullptr, KernelName)),
        SnapshotName(SnapshotName), IType(IType) {
    MnemeSnapshot<VendorTypes>::readMnemeSnapShot(SnapshotName, GlobalVars,
                                                  DeviceMemoryState, KInfo);
    LOG_DEBUG("Initialized Snapshot for kernel {} of state {}",
              KInfo->getName(),
              (IType == InstanceType::Prologue ? "Prologue" : "Epilogue"));
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
        return false;
    }

    for (auto &[GVName, GVI] : GlobalVars) {
      auto it = other.GlobalVars.find(GVName);
      if (it == other.GlobalVars.end()) {
        LOG_WARN("comparing with global var {} that exists only on one of the "
                 "comparators",
                 GVName);
        return false;
      }

      const GlobalVarInfo &OtherGV = it->second;

      std::unique_ptr<uint8_t[]> hostData(new uint8_t[GVI.VarSize]);
      uint8_t *comparator;
      if (IType == InstanceType::Prologue) {
        auto CEC = MnemeDeviceRT::DeviceErrorCheck(
            MnemeDeviceRT::DeviceCopy(hostData.get(), GVI.DevAddr, GVI.VarSize,
                                      MnemeDeviceRT::MemcpyDeviceToHostKind()));
        if (CEC)
          FATAL_ERROR(
              "Could not copy Memory Blob to device EC: " + CEC.value() + "\n");
        comparator = OtherGV.HostAddr.get();
      } else if (other.IType == InstanceType::Prologue) {
        auto CEC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceCopy(
            OtherGV.HostAddr.get(), OtherGV.DevAddr, OtherGV.VarSize,
            MnemeDeviceRT::MemcpyDeviceToHostKind()));
        if (CEC)
          FATAL_ERROR(
              "Could not copy Memory Blob to device EC: " + CEC.value() + "\n");
        comparator = GVI.HostAddr.get();
      } else {
        FATAL_ERROR("Either this or other need to be of Prologue type");
      }

      if (memcmp(comparator, hostData.get(), GVI.VarSize) != 0)
        return false;
    }

    return true;
  }
  // Derive inequality operator
  bool operator!=(const ReplayMemState<VendorTypes> &other) const {
    return !(*this == other);
  }

  std::unique_ptr<void *> getArgs() const {
    void **Args = new void *[KInfo->getNumArgs()];
    auto ArgData = KInfo->getArgData();
    for (int I = 0; I < KInfo->getNumArgs(); I++) {
      Args[I] = ArgData[I].get();
    }
    std::unique_ptr<void *> ArgUniquePtr(Args);
    return ArgUniquePtr;
  }

  void initializeGlobals(DeviceModule_t VendorMod) {
    LOG_INFO("Initializing {} Globals\n", GlobalVars.size());
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
      LOG_INFO("Successfully loaded global variable: {}", KV.first);
    }
  }
};

template <DeviceVendors VendorTypes>
class ReplayInstance : public mneme::KernelInstance {
  using MnemeDeviceRT = DeviceTraits<VendorTypes>;
  using DeviceMemState = ReplayMemState<VendorTypes>;
  using DeviceModule_t = typename DeviceTraits<VendorTypes>::DeviceModule_t;
  std::string KernelName;
  std::string DemangledName;
  void *VAddr;
  uint64_t VASize;
  DeviceMemState PrologueState;
  DeviceMemState EpilogueState;
  llvm::SmallVector<std::string> ModuleFileNames;
  std::unique_ptr<PageManager> PM;

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

    auto VAddrStr = extractStringValue(JSONRoot, "VAddr").str();
    VAddr = util::hexStringToPointer<void *>(VAddrStr);

    LOG_DEBUG("VAddr in json is {}", VAddr);

    auto VASizeOpt = JSONRoot->getInteger("VASize");
    if (!VASizeOpt)
      FATAL_ERROR("Cannot extract Virtual Address Size from JSON DB");
    VASize = *VASizeOpt;
    int DeviceID = 0;
    auto MinPageSize = MnemeDeviceRT::getMinPageSize(DeviceID);

    auto ActualSize = util::roundUp(VASize, MinPageSize);
    if (VASize != ActualSize)
      LOG_WARN("Expected VASize ({}) and ActualSize ({}) to match\n", VASize,
               ActualSize);

    PM = initializePageManager<MnemeDeviceRT>(VAddr, ActualSize);
    if (PM->getVAStart() != VAddr) {
      FATAL_ERROR("Could not allocate Device Pages\n Record got : " +
                  util::pointerToHexString(VAddr) + " and replay got : " +
                  util::pointerToHexString(PM->getVAStart()));
    }

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

    auto ShmSize = InstanceJSON->getInteger("SharedMem");

    if (!ShmSize)
      FATAL_ERROR("Expected Shared Memory Size to be part of the JSON");

    SharedMem = ShmSize.value();

    PrologueState = DeviceMemState(KernelName, PrologueFn,
                                   DeviceMemState::InstanceType::Prologue);
    EpilogueState = DeviceMemState(KernelName, EpilogueFn,
                                   DeviceMemState::InstanceType::Epilogue);
  }

  ~ReplayInstance() {
    PrologueState.release();
    EpilogueState.release();
    MnemeDeviceRT::freeVirtualAddress(PM->getVAStart(), PM->getTotalVASize());
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

  void initializeGlobals(DeviceModule_t VendorMod) {
    for (auto &KV : PrologueState.GlobalVars) {
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
      LOG_INFO("Successfully loaded global variable: {}", KV.first);
    }
  }

  void initializeDeviceMemory() {
    PrologueState.load();
    EpilogueState.load();
  }

  void releaseMemory() { PrologueState.release(); }

  void reset() { PrologueState.reset(); }

  bool isMemorySame() { return PrologueState == EpilogueState; }

  std::unique_ptr<void *> getArgs() const { return PrologueState.getArgs(); }

  uint64_t getSharedMemSize() { return SharedMem; }

  dim3 getRecordedGrid() const { return GridDim; }
  dim3 getRecordedBlock() const { return BlockDim; }
};

} // namespace mneme
