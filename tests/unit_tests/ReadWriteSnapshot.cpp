#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeAnnotation.hpp"
#include "mneme/MnemeKernelInfo.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeReplay.hpp"
#include "mneme/MnemeSnapshot.hpp"
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <memory>
#include <optional>
#include <random>

using namespace mneme;

#ifdef MNEME_ENABLE_HIP
constexpr DeviceVendors Vendor = DeviceVendors::HIP;
using MnemeDeviceRT = DeviceTraits<DeviceVendors::HIP>;
using MnemeMemoryBlobDevice = MnemeMemoryBlob<DeviceVendors::HIP>;
#elif defined(MNEME_ENABLE_CUDA)
constexpr DeviceVendors Vendor = DeviceVendors::CUDA;
using MnemeDeviceRT = DeviceTraits<DeviceVendors::CUDA>;
using MnemeMemoryBlobDevice = MnemeMemoryBlob<DeviceVendors::CUDA>;
#endif

template <typename T> void initializeRandomBuffer(T *Buffer, size_t Size) {
  // Random number generation setup
  std::mt19937 gen(4); // Mersenne Twister random number generator
  std::uniform_int_distribution<T> dis(
      0, 255); // Uniform distribution for char range

  // Fill the buffer with random values
  for (int I = 0; I < Size; I++) {
    Buffer[I] = dis(gen);
  }
}

int main(int argc, char **argv) {
  // We allocate some "fake" globals
  auto initializeDeviceData = [&] {
    uint8_t *HData = new uint8_t[128];
    initializeRandomBuffer(HData, 128);
    uint8_t *DData;

    auto EC = MnemeDeviceRT::DeviceErrorCheck(
        MnemeDeviceRT::DeviceMalloc((void **)&DData, 128));
    if (EC) {
      std::cout << " Here " << EC.value() << "\n";
      LOG_FATAL("Could not allocate device data");
    }

    EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceCopy(
        DData, HData, 128, MnemeDeviceRT::MemcpyHostToDeviceKind()));
    if (EC) {
      std::cout << " Here " << EC.value() << "\n";
      LOG_FATAL("Could not allocate device data");
    }
    return std::make_pair(DData, HData);
  };

  auto BlobData = initializeDeviceData();
  auto GlobalData = initializeDeviceData();

  MnemeMemoryBlobDevice Blob(128L, BlobData.first, 128L);
  mneme::Metadata Md;
  Md.builtin = BuiltinDType::F64;
  Md.norm = Norm::L2;
  Md.threshold = 0.5;
  Md.threshold_kind = ThresholdKind::Relative;
  Md.tag = std::string("Test");
  Blob.setMetadata(Md);

  Blob.setHostData(std::unique_ptr<uint8_t[]>(new uint8_t[128]));

  proteus::runtime::GlobalMetadata GV{GlobalData.second, GlobalData.first, 128};

  std::string KernelName("TestKernel");
  std::shared_ptr<KernelInfo> TestKernel =
      std::make_shared<KernelInfo>(KernelName);

  int NumArgs = 4;
  llvm::SmallVector<size_t> ArgSizes(4);
  initializeRandomBuffer(ArgSizes.data(), 4);
  void **Args = new void *[NumArgs];
  for (auto A = 0; A < NumArgs; A++) {
    Args[A] = reinterpret_cast<void *>(new uint8_t[ArgSizes[A]]);
  }

  TestKernel->setArgSizes(ArgSizes);

  // Create a raw_svector_ostream using the buffer
  proteus::runtime::GlobalMetadataMap GVars;
  GVars.try_emplace("Test", GV);
  llvm::DenseMap<void *, MnemeMemoryBlobDevice> DeviceMemMap;
  DeviceMemMap.try_emplace((void *)BlobData.first, std::move(Blob));
  std::filesystem::path SnapshotFN("./test.mneme");

  MnemeSnapshot<Vendor>::GlobalSnapshotData PrologueGlobals;
  MnemeSnapshot<Vendor>::takeMnemeBytesSnapshot(GVars, DeviceMemMap, SnapshotFN,
                                                TestKernel->KernelArgSizes,
                                                Args, 0, &PrologueGlobals);

  auto ReadSnap =
      MnemeSnapshot<Vendor>::readBytesSnapshot(KernelName, SnapshotFN.string());
  auto &ReadGVars = ReadSnap.GlobalVars;
  auto &ReadDeviceMemMap = ReadSnap.DeviceMemory;
  auto &RTestKernel = ReadSnap.KInfo;

  auto ValidateGlobalMem = [&]() {
    auto it = ReadGVars.find("Test");
    if (it == ReadGVars.end())
      return 2;
    auto &RGV = it->second;

    if (RGV.VarSize != GV.VarSize) {
      std::cerr << "VarSize differs " << RGV.VarSize << " " << GV.VarSize
                << "\n";
      return 2;
    }

    if (std::memcmp(GV.HostAddr, RGV.HostAddr, 128) != 0) {
      std::cerr << "Memory differs between GV and GVR\n";
      return 2;
    }
    std::cerr << "Global Memory is correct\n";
    return 0;
  }();

  auto ValidateDeviceMem = [&]() {
    for (auto &RKV : ReadDeviceMemMap) {
      auto &RBlob = RKV.second;
      if (!DeviceMemMap.contains(RKV.first)) {
        std::cerr << "Address does not exist in Device Map " << std::hex
                  << RKV.first << std::dec << "\n";
        return 1;
      }

      auto &WBlob = DeviceMemMap[RKV.first];

      if (RBlob.getActualSize() != WBlob.getActualSize()) {
        std::cerr << "Actual Sizes differ " << RBlob.getActualSize() << " "
                  << WBlob.getActualSize() << "\n";
        return 1;
      }

      if (RBlob.getSize() != WBlob.getSize()) {
        std::cerr << "Sizes differ" << WBlob.getSize() << " " << RBlob.getSize()
                  << "\n";
        return 1;
      }

      if (RBlob.getMetadata().builtin != BuiltinDType::F64) {
        std::cerr << "Metadata builtin differs\n";
        return 1;
      }

      if (RBlob.getMetadata().norm != Norm::L2) {
        std::cerr << "Metadata norm differs\n";
        return 1;
      }

      if (RBlob.getMetadata().threshold != 0.5) {
        std::cerr << "Metadata threshold differs\n";
        return 1;
      }

      if (RBlob.getMetadata().threshold_kind != ThresholdKind::Relative) {
        std::cerr << "Metadata threshold_kind differs\n";
        return 1;
      }

      if (RBlob.getMetadata().tag.value() != "Test") {
        std::cerr << "Metadata tag differs\n";
        return 1;
      }

      uint8_t *WData = WBlob.getHostData().get();
      uint8_t *RData = RBlob.getHostData().get();
      if (std::memcmp(reinterpret_cast<void *>(WData),
                      reinterpret_cast<void *>(RData), 128) != 0) {
        std::cerr << "Memory differs between GV and GVR\n";
        return 1;
      }
    }
    return 0;
  }();

  auto ValidateKernelArgs = [&]() {
    auto &WKernel = *TestKernel;
    auto &RKernel = *RTestKernel;
    if (WKernel.getNumArgs() != RKernel.getNumArgs()) {
      std::cerr << "Number of recorded arguments differ "
                << WKernel.getNumArgs() << " and read " << RKernel.getNumArgs()
                << "\n";
      return 4;
    }

    auto WArgSizes = WKernel.getArgSizes();
    auto RArgSizes = RKernel.getArgSizes();
    for (auto A = 0; A < WKernel.getNumArgs(); A++) {
      if (WArgSizes[A] != RArgSizes[A]) {
        std::cerr << "The size of argument " << A
                  << " differs WAS:" << WArgSizes[A] << " RAS:" << RArgSizes[A]
                  << "\n";
        return 4;
      }
    }

    auto WArgData = WKernel.getArgData();
    auto RArgData = RKernel.getArgData();
    for (auto A = 0; A < WKernel.getNumArgs(); A++) {
      if (std::memcmp(Args[A], RArgData[A].get(), WArgSizes[A]) != 0) {
        if (WArgSizes[A] != RArgSizes[A]) {
          std::cerr << "The Memory of argument " << A << "differs \n";
          return 4;
        }
      }
    }
    return 0;
  }();

  // make some changes to the device memory and global memory to test diffing
  BlobData.second[0] ^= 0x7;
  BlobData.second[63] ^= 0x11;
  BlobData.second[64] ^= 0x23;
  BlobData.second[127] ^= 0x42;
  GlobalData.second[2] ^= 0x5;
  GlobalData.second[3] ^= 0x9;
  GlobalData.second[4] ^= 0x13;

  auto EC = MnemeDeviceRT::DeviceErrorCheck(
      MnemeDeviceRT::DeviceCopy(BlobData.first, BlobData.second, 128,
                                MnemeDeviceRT::MemcpyHostToDeviceKind()));
  if (EC)
    LOG_FATAL("Could not update device blob data");

  EC = MnemeDeviceRT::DeviceErrorCheck(
      MnemeDeviceRT::DeviceCopy(GlobalData.first, GlobalData.second, 128,
                                MnemeDeviceRT::MemcpyHostToDeviceKind()));
  if (EC)
    LOG_FATAL("Could not update device global data");

  std::filesystem::path DiffSnapshotFN("./test.epilogue.mneme");
  MnemeSnapshot<Vendor>::takeMnemeDiffSnapshot(
      GVars, DeviceMemMap, DiffSnapshotFN, PrologueGlobals, 0);

  auto DiffSnap = MnemeSnapshot<Vendor>::readDiffSnapshot(
      KernelName, DiffSnapshotFN.string(), SnapshotFN.string());
  auto &DiffGVars = DiffSnap.GlobalVars;
  auto &DiffDeviceMemMap = DiffSnap.DeviceMemory;
  auto &DiffKernel = DiffSnap.KInfo;

  std::string DiffPlanError;
  auto DiffPlan = MnemeSnapshot<Vendor>::readDiffPlan(DiffSnapshotFN.string(),
                                                      &DiffPlanError);
  auto ValidateDiffPlan = [&]() {
    if (!DiffPlan) {
      std::cerr << "Could not parse diff plan: " << DiffPlanError << "\n";
      return 64;
    }
    if (DiffPlan->Globals.size() != 1 || DiffPlan->Blobs.size() != 1) {
      std::cerr << "Diff plan counts differ\n";
      return 64;
    }
    if (DiffPlan->RawRangeCount == 0 || DiffPlan->ChangedBytes == 0) {
      std::cerr << "Diff plan did not record changed ranges\n";
      return 64;
    }
    if (DiffPlan->Globals[0].Ranges.empty() ||
        DiffPlan->Blobs[0].Ranges.empty()) {
      std::cerr << "Diff plan missing global or blob ranges\n";
      return 64;
    }
    return 0;
  }();

  auto ValidateDiffGlobalMem = [&]() {
    auto it = DiffGVars.find("Test");
    if (it == DiffGVars.end())
      return 8;
    auto &RGV = it->second;

    if (RGV.VarSize != GV.VarSize) {
      std::cerr << "Diff VarSize differs " << RGV.VarSize << " " << GV.VarSize
                << "\n";
      return 8;
    }

    if (std::memcmp(GlobalData.second, RGV.HostAddr, 128) != 0) {
      std::cerr << "Diff global memory did not reconstruct epilogue data\n";
      return 8;
    }
    return 0;
  }();

  auto ValidateDiffDeviceMem = [&]() {
    auto it = DiffDeviceMemMap.find((void *)BlobData.first);
    if (it == DiffDeviceMemMap.end()) {
      std::cerr << "Diff device map is missing blob address\n";
      return 16;
    }

    auto &RBlob = it->second;
    if (RBlob.getActualSize() != 128 || RBlob.getSize() != 128) {
      std::cerr << "Diff blob sizes differ\n";
      return 16;
    }

    if (RBlob.getMetadata().builtin != BuiltinDType::F64 ||
        RBlob.getMetadata().norm != Norm::L2 ||
        RBlob.getMetadata().threshold != 0.5 ||
        RBlob.getMetadata().threshold_kind != ThresholdKind::Relative ||
        RBlob.getMetadata().tag.value() != "Test") {
      std::cerr << "Diff blob metadata differs\n";
      return 16;
    }

    if (std::memcmp(BlobData.second, RBlob.getHostData().get(), 128) != 0) {
      std::cerr << "Diff device memory did not reconstruct epilogue data\n";
      return 16;
    }
    return 0;
  }();

  auto ValidateDiffKernelArgs = [&]() {
    if (TestKernel->getNumArgs() != DiffKernel->getNumArgs()) {
      std::cerr << "Diff snapshot did not inherit prologue arguments\n";
      return 32;
    }

    auto WArgSizes = TestKernel->getArgSizes();
    auto RArgSizes = DiffKernel->getArgSizes();
    auto RArgData = DiffKernel->getArgData();
    for (auto A = 0; A < TestKernel->getNumArgs(); A++) {
      if (WArgSizes[A] != RArgSizes[A] ||
          std::memcmp(Args[A], RArgData[A].get(), WArgSizes[A]) != 0) {
        std::cerr << "Diff snapshot argument " << A
                  << " differs from prologue\n";
        return 32;
      }
    }
    return 0;
  }();

  auto ValidateReplayDiffReset = [&]() {
    PrologueState<Vendor> PrologueReplay(KernelName, SnapshotFN.string());
    EpilogueState<Vendor> EpilogueReplay(
        MnemeSnapshot<Vendor>::readDiffSnapshot(
            KernelName, DiffSnapshotFN.string(), SnapshotFN.string()),
        DiffSnapshotFN.string());

    PrologueReplay.load();

    auto CheckMode = [&](ReplayResetMode Mode) {
      PrologueReplay.prepareResetPlan(EpilogueReplay, Mode);

      auto EC = MnemeDeviceRT::DeviceErrorCheck(
          MnemeDeviceRT::DeviceCopy(BlobData.first, BlobData.second, 128,
                                    MnemeDeviceRT::MemcpyHostToDeviceKind()));
      if (EC)
        LOG_FATAL("Could not poison reset device blob data");

      EC = MnemeDeviceRT::DeviceErrorCheck(
          MnemeDeviceRT::DeviceCopy(GlobalData.first, GlobalData.second, 128,
                                    MnemeDeviceRT::MemcpyHostToDeviceKind()));
      if (EC)
        LOG_FATAL("Could not poison reset global data");

      PrologueReplay.reset(Mode);

      std::unique_ptr<uint8_t[]> DeviceCheck(new uint8_t[128]);
      EC = MnemeDeviceRT::DeviceErrorCheck(
          MnemeDeviceRT::DeviceCopy(DeviceCheck.get(), BlobData.first, 128,
                                    MnemeDeviceRT::MemcpyDeviceToHostKind()));
      if (EC)
        LOG_FATAL("Could not copy reset device blob data");

      auto BlobIt = ReadDeviceMemMap.find((void *)BlobData.first);
      if (BlobIt == ReadDeviceMemMap.end() ||
          std::memcmp(DeviceCheck.get(), BlobIt->second.getHostData().get(),
                      128) != 0) {
        std::cerr << "Diff reset did not restore prologue blob data\n";
        return 128;
      }

      EC = MnemeDeviceRT::DeviceErrorCheck(
          MnemeDeviceRT::DeviceCopy(DeviceCheck.get(), GlobalData.first, 128,
                                    MnemeDeviceRT::MemcpyDeviceToHostKind()));
      if (EC)
        LOG_FATAL("Could not copy reset global data");

      auto GlobalIt = ReadGVars.find("Test");
      if (GlobalIt == ReadGVars.end() ||
          std::memcmp(DeviceCheck.get(), GlobalIt->second.HostAddr, 128) != 0) {
        std::cerr << "Diff reset did not restore prologue global data\n";
        return 128;
      }

      return 0;
    };

    auto DiffRet = CheckMode(ReplayResetMode::Diff);

    PrologueReplay.release();
    return DiffRet;
  }();

  auto Ret = ValidateGlobalMem | ValidateDeviceMem | ValidateKernelArgs |
             ValidateDiffGlobalMem | ValidateDiffDeviceMem |
             ValidateDiffKernelArgs | ValidateDiffPlan |
             ValidateReplayDiffReset;

  delete[] GlobalData.second;
  delete[] BlobData.second;

  EC = MnemeDeviceRT::DeviceErrorCheck(
      MnemeDeviceRT::DeviceFree(GlobalData.first));
  if (EC)
    LOG_FATAL("Could not release device memory\n");

  EC = MnemeDeviceRT::DeviceErrorCheck(
      MnemeDeviceRT::DeviceFree(BlobData.first));
  if (EC)
    LOG_FATAL("Could not release device memory\n");

  return Ret;
}
