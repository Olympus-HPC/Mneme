#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeAnnotation.hpp"
#include "mneme/MnemeKernelInfo.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeSnapshot.hpp"
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
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

  constexpr uint64_t BlobId = 7;
  constexpr uint64_t BlobOffset = 4096;
  MnemeMemoryBlobDevice Blob(128L, BlobData.first, 128L, BlobId, BlobOffset);
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

  // The last argument is a pointer 16 bytes into the blob and must round trip
  // as a managed pointer rather than as raw bytes.
  constexpr int PointerArg = 4;
  constexpr uint64_t PointerArgOffset = 16;
  int NumArgs = 5;
  llvm::SmallVector<size_t> ArgSizes(NumArgs);
  initializeRandomBuffer(ArgSizes.data(), 4);
  ArgSizes[PointerArg] = sizeof(void *);
  void **Args = new void *[NumArgs];
  for (auto A = 0; A < NumArgs; A++) {
    Args[A] = reinterpret_cast<void *>(new uint8_t[ArgSizes[A]]);
  }
  void *PointerArgValue = BlobData.first + PointerArgOffset;
  std::memcpy(Args[PointerArg], &PointerArgValue, sizeof(PointerArgValue));

  TestKernel->setArgSizes(ArgSizes);

  // Create a raw_svector_ostream using the buffer
  proteus::runtime::GlobalMetadataMap GVars;
  GVars.try_emplace("Test", GV);
  llvm::DenseMap<void *, MnemeMemoryBlobDevice> DeviceMemMap;
  DeviceMemMap.try_emplace((void *)BlobData.first, std::move(Blob));
  std::filesystem::path SnapshotFN("./test.mneme");

  auto PrologueGlobals = std::make_shared<GlobalSnapshotData>();
  SnapshotInput<Vendor> In{GVars, DeviceMemMap, TestKernel->KernelArgSizes,
                           Args, nullptr};
  BytesWriter<Vendor> PrologueWriter(PrologueGlobals);
  size_t MeasuredBytes = PrologueWriter.measure(In);
  PrologueWriter.write(SnapshotFN, In);

  auto ReadSnap = SnapshotFormatRegistry<Vendor>::open(SnapshotFN.string())
                      ->read(KernelName, BaseSnapshotSource<Vendor>());
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
    if (!ReadSnap.RelocatableBlobs) {
      std::cerr << "Bytes snapshot should have relocatable blobs\n";
      return 1;
    }
    for (auto &RKV : ReadDeviceMemMap) {
      auto &RBlob = RKV.second;
      if (RKV.first != BlobId || RBlob.getBlobId() != BlobId ||
          RBlob.getBlobOffset() != BlobOffset) {
        std::cerr << "Blob id or offset differs\n";
        return 1;
      }

      auto &WBlob = DeviceMemMap[(void *)BlobData.first];

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

    auto RArgData = RKernel.getArgData();
    auto RKinds = RKernel.getArgEncodingKinds();
    for (auto A = 0; A < WKernel.getNumArgs(); A++) {
      if (A == PointerArg) {
        if (RKinds[A] != KernelArgEncodingKind::ManagedPointer ||
            RKernel.getManagedArgBlobId(A) != BlobId ||
            RKernel.getManagedArgOffset(A) != PointerArgOffset) {
          std::cerr << "Pointer argument was not stored as a managed pointer\n";
          return 4;
        }
        continue;
      }
      if (RKinds[A] != KernelArgEncodingKind::RawBytes ||
          std::memcmp(Args[A], RArgData[A].get(), WArgSizes[A]) != 0) {
        std::cerr << "The Memory of argument " << A << " differs \n";
        return 4;
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

  auto EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceCopy(
      BlobData.first, BlobData.second, 128,
      MnemeDeviceRT::MemcpyHostToDeviceKind()));
  if (EC)
    LOG_FATAL("Could not update device blob data");

  EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceCopy(
      GlobalData.first, GlobalData.second, 128,
      MnemeDeviceRT::MemcpyHostToDeviceKind()));
  if (EC)
    LOG_FATAL("Could not update device global data");

  std::filesystem::path DiffSnapshotFN("./test.epilogue.mneme");
  DiffWriter<Vendor> DiffSnapshotWriter(PrologueGlobals);
  size_t MeasuredDiff = DiffSnapshotWriter.measure(In);
  DiffSnapshotWriter.write(DiffSnapshotFN, In);

  auto DiffSnap =
      SnapshotFormatRegistry<Vendor>::open(DiffSnapshotFN.string())
          ->read(KernelName, BaseSnapshotSource<Vendor>(SnapshotFN.string()));
  auto &DiffGVars = DiffSnap.GlobalVars;
  auto &DiffDeviceMemMap = DiffSnap.DeviceMemory;
  auto &DiffKernel = DiffSnap.KInfo;

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
    auto it = DiffDeviceMemMap.find(BlobId);
    if (it == DiffDeviceMemMap.end()) {
      std::cerr << "Diff device map is missing blob id\n";
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
      if (A == PointerArg)
        continue;
      if (WArgSizes[A] != RArgSizes[A] ||
          std::memcmp(Args[A], RArgData[A].get(), WArgSizes[A]) != 0) {
        std::cerr << "Diff snapshot argument " << A
                  << " differs from prologue\n";
        return 32;
      }
    }
    return 0;
  }();

  auto ResetBlobBase = [&]() {
    auto PrologueBlobIt = ReadDeviceMemMap.find(BlobId);
    auto HostData = std::unique_ptr<uint8_t[]>(new uint8_t[128]);
    std::memcpy(HostData.get(), PrologueBlobIt->second.getHostData().get(),
                128);
    DeviceMemMap[(void *)BlobData.first].setHostData(std::move(HostData));
  };

  llvm::SmallVector<size_t> EmptyArgSizes;
  SnapshotInput<Vendor> InNoArgs{GVars, DeviceMemMap, EmptyArgSizes, nullptr,
                                 nullptr};

  ResetBlobBase();
  std::filesystem::path SparseBestSnapshotFN("./test.best.sparse.mneme");
  BestWriter<Vendor>(PrologueGlobals).write(SparseBestSnapshotFN, InNoArgs);
  auto ValidateBestSparse = [&]() {
    if (!SnapshotFormatRegistry<Vendor>::open(SparseBestSnapshotFN.string())
             ->requiresBaseSnapshot()) {
      std::cerr << "Best sparse snapshot should choose diff\n";
      return 64;
    }
    auto BestSparseSnap =
        SnapshotFormatRegistry<Vendor>::open(SparseBestSnapshotFN.string())
            ->read(KernelName, BaseSnapshotSource<Vendor>(SnapshotFN.string()));
    auto It = BestSparseSnap.DeviceMemory.find(BlobId);
    if (It == BestSparseSnap.DeviceMemory.end() ||
        std::memcmp(BlobData.second, It->second.getHostData().get(), 128) !=
            0) {
      std::cerr << "Best sparse snapshot did not reconstruct epilogue data\n";
      return 64;
    }
    return 0;
  }();

  auto PrologueBlobIt = ReadDeviceMemMap.find(BlobId);
  auto PrologueGlobalIt = ReadGVars.find("Test");
  auto *PrologueBlob = PrologueBlobIt->second.getHostData().get();
  auto *PrologueGlobal =
      static_cast<uint8_t *>(PrologueGlobalIt->second.HostAddr);
  // Alternating changes force many one-byte diff ranges, making the bytes
  // snapshot smaller than the diff snapshot.
  for (size_t I = 0; I < 128; ++I) {
    BlobData.second[I] = (I % 2 == 0) ? (PrologueBlob[I] ^ 0xff)
                                      : PrologueBlob[I];
    GlobalData.second[I] = (I % 2 == 0) ? (PrologueGlobal[I] ^ 0xff)
                                        : PrologueGlobal[I];
  }

  EC = MnemeDeviceRT::DeviceErrorCheck(
      MnemeDeviceRT::DeviceCopy(BlobData.first, BlobData.second, 128,
                                MnemeDeviceRT::MemcpyHostToDeviceKind()));
  if (EC)
    LOG_FATAL("Could not update fragmented device blob data");

  EC = MnemeDeviceRT::DeviceErrorCheck(
      MnemeDeviceRT::DeviceCopy(GlobalData.first, GlobalData.second, 128,
                                MnemeDeviceRT::MemcpyHostToDeviceKind()));
  if (EC)
    LOG_FATAL("Could not update fragmented device global data");

  ResetBlobBase();
  std::filesystem::path FragmentedBestSnapshotFN(
      "./test.best.fragmented.mneme");
  BestWriter<Vendor>(PrologueGlobals).write(FragmentedBestSnapshotFN, InNoArgs);
  auto ValidateBestFragmented = [&]() {
    if (SnapshotFormatRegistry<Vendor>::open(FragmentedBestSnapshotFN.string())
            ->requiresBaseSnapshot()) {
      std::cerr << "Best fragmented snapshot should choose bytes\n";
      return 128;
    }
    auto BestFragmentedSnap =
        SnapshotFormatRegistry<Vendor>::open(FragmentedBestSnapshotFN.string())
            ->read(KernelName, BaseSnapshotSource<Vendor>(SnapshotFN.string()));
    auto It = BestFragmentedSnap.DeviceMemory.find(BlobId);
    if (It == BestFragmentedSnap.DeviceMemory.end() ||
        std::memcmp(BlobData.second, It->second.getHostData().get(), 128) !=
            0) {
      std::cerr
          << "Best fragmented snapshot did not reconstruct epilogue data\n";
      return 128;
    }
    return 0;
  }();

  // "Best" mode decides on measure(), so it must match the written size.
  auto ValidateMeasure = [&]() {
    auto ActualBytes = std::filesystem::file_size(SnapshotFN);
    if (MeasuredBytes != ActualBytes) {
      std::cerr << "Bytes snapshot measured " << MeasuredBytes << " but wrote "
                << ActualBytes << "\n";
      return 512;
    }

    auto ActualDiff = std::filesystem::file_size(DiffSnapshotFN);
    if (MeasuredDiff != ActualDiff) {
      std::cerr << "Diff snapshot measured " << MeasuredDiff << " but wrote "
                << ActualDiff << "\n";
      return 512;
    }
    return 0;
  }();

  // Pin all three prefixes SnapshotHeader::parse recognises.
  auto ValidateHeaderParse = [&]() {
    char Container[SnapshotHeader::Size];
    std::memcpy(Container, SnapshotHeader::Magic,
                sizeof(SnapshotHeader::Magic));
    uint32_t Kind = 2;
    uint32_t Version = 7;
    std::memcpy(Container + 8, &Kind, sizeof(Kind));
    std::memcpy(Container + 12, &Version, sizeof(Version));
    auto [ContainerHeader, ContainerOffset] =
        SnapshotHeader::parse(llvm::StringRef(Container, sizeof(Container)));
    if (ContainerHeader.Kind != SnapshotKind::Diff ||
        ContainerHeader.Version != 7 || ContainerOffset != 16) {
      std::cerr << "Container prefix did not parse as {Diff, 7, 16}\n";
      return 256;
    }

    std::string Legacy("MNEME_DIFF_V1");
    Legacy += "arbitrary diff payload";
    auto [LegacyHeader, LegacyOffset] = SnapshotHeader::parse(Legacy);
    if (LegacyHeader.Kind != SnapshotKind::Diff || LegacyHeader.Version != 1 ||
        LegacyOffset != 13) {
      std::cerr << "Legacy diff magic did not parse as {Diff, 1, 13}\n";
      return 256;
    }

    std::string Headerless("arbitrary bytes snapshot payload");
    auto [BytesHeader, BytesOffset] = SnapshotHeader::parse(Headerless);
    if (BytesHeader.Kind != SnapshotKind::Bytes || BytesHeader.Version != 0 ||
        BytesOffset != 0) {
      std::cerr << "Unprefixed buffer did not parse as {Bytes, 0, 0}\n";
      return 256;
    }

    auto parseFilePrefix = [](const std::filesystem::path &Path) {
      std::ifstream FileIn(Path, std::ios::binary);
      std::string Prefix(SnapshotHeader::Size, '\0');
      FileIn.read(Prefix.data(), Prefix.size());
      return SnapshotHeader::parse(Prefix);
    };

    auto [PrologueHeader, PrologueOffset] = parseFilePrefix(SnapshotFN);
    if (PrologueHeader.Kind != SnapshotKind::Bytes ||
        PrologueHeader.Version != 1 || PrologueOffset != 16) {
      std::cerr << "Prologue file did not parse as {Bytes, 1, 16}\n";
      return 256;
    }

    auto [DiffFileHeader, DiffFileOffset] = parseFilePrefix(DiffSnapshotFN);
    if (DiffFileHeader.Kind != SnapshotKind::Diff ||
        DiffFileHeader.Version != 2 || DiffFileOffset != 16) {
      std::cerr << "Diff file did not parse as {Diff, 2, 16}\n";
      return 256;
    }
    return 0;
  }();

  // Recordings made before blobs had ids must still open. Build the old
  // layouts by hand: a headerless bytes prologue keyed by device address with
  // raw arguments, and a diff with the legacy magic on top of it.
  auto ValidateLegacyLayouts = [&]() {
    std::filesystem::path LegacyPrologueFN("./test.legacy.mneme");
    std::filesystem::path LegacyDiffFN("./test.legacy.epilogue.mneme");
    std::string GlobalName("Test");
    size_t One = 1;
    size_t Zero = 0;
    uint64_t LegacyPointerArg = 0xdeadbeef;

    {
      std::error_code EC;
      llvm::raw_fd_ostream OS(LegacyPrologueFN.string(), EC);
      util::writeScalar(OS, One);
      GlobalVarHeader{GlobalName, 128, GlobalData.first}.write(OS);
      util::writeBytes(OS, llvm::ArrayRef<uint8_t>(PrologueGlobal, 128));
      util::writeScalar(OS, One);
      util::writeScalar(OS, size_t{128});
      util::writeScalar(OS, size_t{128});
      util::writeScalar(OS, BlobData.first);
      util::writeBytes(OS, llvm::ArrayRef<uint8_t>(PrologueBlob, 128));
      mneme::metadata::serialize(OS, Md);
      util::writeScalar(OS, One);
      util::writeScalar(OS, sizeof(LegacyPointerArg));
      util::writeScalar(OS, LegacyPointerArg);
    }

    {
      std::error_code EC;
      llvm::raw_fd_ostream OS(LegacyDiffFN.string(), EC);
      util::writeBytes(OS, llvm::StringRef("MNEME_DIFF_V1"));
      util::writeScalar(OS, One);
      GlobalVarHeader{GlobalName, 128, GlobalData.first}.write(OS);
      util::writeScalar(OS, One);
      util::writeScalar(OS, size_t{2});
      util::writeScalar(OS, size_t{3});
      util::writeBytes(OS, llvm::ArrayRef<uint8_t>(GlobalData.second + 2, 3));
      util::writeScalar(OS, One);
      util::writeScalar(OS, size_t{128});
      util::writeScalar(OS, size_t{128});
      util::writeScalar(OS, BlobData.first);
      mneme::metadata::serialize(OS, Md);
      util::writeScalar(OS, One);
      util::writeScalar(OS, Zero);
      util::writeScalar(OS, size_t{128});
      util::writeBytes(OS, llvm::ArrayRef<uint8_t>(BlobData.second, 128));
    }

    auto LegacyId = reinterpret_cast<uint64_t>(BlobData.first);
    auto LegacySnap =
        SnapshotFormatRegistry<Vendor>::open(LegacyPrologueFN.string())
            ->read(KernelName, BaseSnapshotSource<Vendor>());
    if (LegacySnap.RelocatableBlobs) {
      std::cerr << "Legacy bytes snapshot should not be relocatable\n";
      return 1024;
    }
    auto LegacyBlobIt = LegacySnap.DeviceMemory.find(LegacyId);
    if (LegacyBlobIt == LegacySnap.DeviceMemory.end() ||
        LegacyBlobIt->second.getBlobAddr() != nullptr ||
        std::memcmp(PrologueBlob, LegacyBlobIt->second.getHostData().get(),
                    128) != 0 ||
        LegacyBlobIt->second.getMetadata().tag.value() != "Test") {
      std::cerr << "Legacy bytes snapshot did not key the blob by address\n";
      return 1024;
    }
    auto &LegacyKernel = *LegacySnap.KInfo;
    if (LegacyKernel.getNumArgs() != 1 ||
        LegacyKernel.getArgEncodingKinds()[0] !=
            KernelArgEncodingKind::RawBytes ||
        std::memcmp(LegacyKernel.getArgData()[0].get(), &LegacyPointerArg,
                    sizeof(LegacyPointerArg)) != 0) {
      std::cerr << "Legacy bytes snapshot did not keep raw arguments\n";
      return 1024;
    }

    auto LegacyDiffSnap =
        SnapshotFormatRegistry<Vendor>::open(LegacyDiffFN.string())
            ->read(KernelName,
                   BaseSnapshotSource<Vendor>(LegacyPrologueFN.string()));
    if (LegacyDiffSnap.RelocatableBlobs) {
      std::cerr << "Legacy diff snapshot should not be relocatable\n";
      return 1024;
    }
    auto LegacyDiffBlobIt = LegacyDiffSnap.DeviceMemory.find(LegacyId);
    auto LegacyDiffGlobalIt = LegacyDiffSnap.GlobalVars.find(GlobalName);
    if (LegacyDiffBlobIt == LegacyDiffSnap.DeviceMemory.end() ||
        std::memcmp(BlobData.second,
                    LegacyDiffBlobIt->second.getHostData().get(), 128) != 0 ||
        LegacyDiffGlobalIt == LegacyDiffSnap.GlobalVars.end() ||
        std::memcmp(GlobalData.second, LegacyDiffGlobalIt->second.HostAddr,
                    128) != 0) {
      std::cerr << "Legacy diff snapshot did not reconstruct epilogue data\n";
      return 1024;
    }
    return 0;
  }();

  auto Ret = ValidateGlobalMem | ValidateDeviceMem | ValidateKernelArgs |
             ValidateDiffGlobalMem | ValidateDiffDeviceMem |
             ValidateDiffKernelArgs | ValidateBestSparse |
             ValidateBestFragmented | ValidateMeasure | ValidateHeaderParse |
             ValidateLegacyLayouts;

  delete[] GlobalData.second;
  delete[] BlobData.second;

  EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceFree(GlobalData.first));
  if (EC)
    LOG_FATAL("Could not release device memory\n");

  EC = MnemeDeviceRT::DeviceErrorCheck(
      MnemeDeviceRT::DeviceFree(BlobData.first));
  if (EC)
    LOG_FATAL("Could not release device memory\n");

  return Ret;
}
