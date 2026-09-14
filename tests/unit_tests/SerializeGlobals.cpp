#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeSnapshot.hpp"
#include <cstdint>
#include <random>

using namespace mneme;

#ifdef MNEME_ENABLE_HIP
using MnemeDeviceRT = DeviceTraits<DeviceVendors::HIP>;
constexpr auto Vendor = DeviceVendors::HIP;
#elif defined(MNEME_ENABLE_CUDA)
using MnemeDeviceRT = DeviceTraits<DeviceVendors::CUDA>;
constexpr auto Vendor = DeviceVendors::CUDA;
#endif

void initializeRandomBuffer(uint8_t *Buffer, size_t Size) {
  // Random number generation setup
  std::mt19937 gen(4); // Mersenne Twister random number generator
  std::uniform_int_distribution<uint8_t> dis(
    0, 255); // Uniform distribution for char range

  // Fill the buffer with random values
  for (int I = 0; I < Size; I++) {
    Buffer[I] = dis(gen);
  }
}

int main(int argc, char **argv) {
  // We allocate some "fake" globals
  uint8_t *HData = new uint8_t[128];
  initializeRandomBuffer(HData, 128);
  void *DData;

  auto EC = MnemeDeviceRT::DeviceErrorCheck(
    MnemeDeviceRT::DeviceMalloc((void **)&DData, 128));
  if (EC)
    LOG_FATAL("Could not allocate device data");

  EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceCopy(
    DData, HData, 128, MnemeDeviceRT::MemcpyHostToDeviceKind()));
  if (EC)
    LOG_FATAL("Could not allocate device data");

  proteus::runtime::GlobalMetadata GV{HData, DData, 128};
  llvm::SmallVector<char, 128> Buffer;
  std::string VarName("Test");

  // Create a raw_svector_ostream using the buffer
  llvm::raw_svector_ostream Stream(Buffer);
  auto StrLen = VarName.size();
  Stream << llvm::StringRef(reinterpret_cast<const char *>(&StrLen),
                            sizeof(StrLen));
  Stream << VarName;
  Stream << llvm::StringRef(reinterpret_cast<const char *>(&GV.VarSize),
                            sizeof(GV.VarSize));
  Stream << llvm::StringRef(reinterpret_cast<const char *>(&GV.DevAddr),
                            sizeof(GV.DevAddr));
  Stream << llvm::StringRef(reinterpret_cast<const char *>(GV.HostAddr),
                            GV.VarSize);

  auto Buff = const_cast<const char *>(Buffer.data());
  auto tmp = mneme::readGlobalVarRecord(Buff);
  auto &GName = tmp.first;
  auto &GVR   = tmp.second;

  auto Ret = [&]() {
    if (GName != VarName) {
      std::cerr << "Global Variables differ in name " << GName << " "
        << GName << "\n";
      return -1;
    }

    if (GVR.VarSize != GV.VarSize) {
      std::cerr << "VarSize differs " << GVR.VarSize << " " << GV.VarSize
        << "\n";
      return -1;
    }

    if (std::memcmp(GVR.HostAddr, GV.HostAddr, 128) != 0) {
      std::cerr << "Memory differs between GV and GVR\n";
      return -1;
    }
    return 0;
  }();

  delete[] HData;
  HData = nullptr;

  EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceFree(DData));
  if (EC)
    LOG_FATAL("Could not release device memory\n");

  return Ret;
}
