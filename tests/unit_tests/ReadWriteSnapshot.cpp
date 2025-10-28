#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeKernelInfo.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeSnapshot.hpp"
#include "mneme/MnemeSymbols.hpp"
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <memory>
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
    if (EC)
      LOG_FATAL("Could not allocate device data");

    EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceCopy(
        DData, HData, 128, MnemeDeviceRT::MemcpyHostToDeviceKind()));
    if (EC)
      LOG_FATAL("Could not allocate device data");
    return std::make_pair(DData, HData);
  };

  auto BlobData = initializeDeviceData();
  auto GlobalData = initializeDeviceData();

  MnemeMemoryBlobDevice Blob(128L, BlobData.first, 128L);

  Blob.setHostData(std::unique_ptr<uint8_t[]>(new uint8_t[128]));

  GlobalVarInfo GV("Test", nullptr, 128, GlobalData.first);
  std::memcpy(GV.HostAddr.get(), GlobalData.second, 128);

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
  llvm::SmallVector<GlobalVarInfo> GVars{GV};
  llvm::DenseMap<void *, MnemeMemoryBlobDevice> DeviceMemMap;
  DeviceMemMap.try_emplace((void *)BlobData.first, std::move(Blob));
  std::filesystem::path SnapshotFN("./test.mneme");

  MnemeSnapshot<Vendor>::takeMnemeSnapshot(GVars, DeviceMemMap, SnapshotFN,
                                           TestKernel->KernelArgSizes, Args, 0);

  llvm::DenseMap<std::string, GlobalVarInfo> ReadGVars;
  llvm::DenseMap<void *, MnemeMemoryBlobDevice> ReadDeviceMemMap;
  std::string RKernelName("TestKernel");
  std::shared_ptr<KernelInfo> RTestKernel =
      std::make_shared<KernelInfo>(KernelName);

  MnemeSnapshot<Vendor>::readMnemeSnapShot(SnapshotFN, ReadGVars,
                                           ReadDeviceMemMap, RTestKernel);

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

  auto ValidateGlobalMem = [&]() {
    auto it = ReadGVars.find(GV.Name);
    if (it == ReadGVars.end())
      return 2;
    auto &RGV = it->second;

    if (RGV.Name != GV.Name) {
      std::cerr << "Global Variables differ in name " << RGV.Name << " "
                << GV.Name << "\n";
      return 2;
    }

    if (RGV.VarSize != GV.VarSize) {
      std::cerr << "VarSize differs " << RGV.VarSize << " " << GV.VarSize
                << "\n";
      return 2;
    }

    if (std::memcmp(GV.HostAddr.get(), RGV.HostAddr.get(), 128) != 0) {
      std::cerr << "Memory differs between GV and GVR\n";
      return 2;
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

  auto Ret = ValidateGlobalMem | ValidateDeviceMem | ValidateKernelArgs;

  delete[] GlobalData.second;
  delete[] BlobData.second;

  auto EC = MnemeDeviceRT::DeviceErrorCheck(
      MnemeDeviceRT::DeviceFree(GlobalData.first));
  if (EC)
    LOG_FATAL("Could not release device memory\n");

  EC = MnemeDeviceRT::DeviceErrorCheck(
      MnemeDeviceRT::DeviceFree(BlobData.first));
  if (EC)
    LOG_FATAL("Could not release device memory\n");

  return Ret;
}
