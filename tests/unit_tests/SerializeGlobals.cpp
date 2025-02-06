#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeSymbols.hpp"
#include <cstdint>
#include <random>

#ifdef ENABLE_HIP
#include "mneme/MnemeMemoryHIP.hpp"
#include "mneme/MnemeRecordHIP.hpp"
#endif

using namespace mneme;

using MnemeRecorderDevice = MnemeRecorderHIP;
using MnemeMemoryBlobDevice = MnemeMemoryBlob<DeviceVendors::HIP>;
using Vendor = DeviceTraits<DeviceVendors::HIP>;

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
  uint8_t *DData;

  auto EC =
      Vendor::DeviceErrorCheck(Vendor::DeviceMalloc((void **)&DData, 128));
  if (EC)
    FATAL_ERROR("Could not allocate device data");

  EC = Vendor::DeviceErrorCheck(
      Vendor::DeviceCopy(DData, HData, 128, Vendor::MemcpyHostToDeviceKind()));
  if (EC)
    FATAL_ERROR("Could not allocate device data");

  GlobalVarInfo GV("Test", nullptr, 128, DData);
  std::memcpy(GV.HostAddr.get(), HData, 128);

  llvm::SmallVector<char, 128> Buffer;

  // Create a raw_svector_ostream using the buffer
  llvm::raw_svector_ostream Stream(Buffer);
  Stream << GV;
  auto Buff = const_cast<const char *>(Buffer.data());
  GlobalVarInfo GVR = GlobalVarInfo::fromBuffer(Buff);

  auto Ret = [&]() {
    if (GVR.Name != GV.Name) {
      std::cerr << "Global Variables differ in name " << GVR.Name << " "
                << GV.Name << "\n";
      return -1;
    }

    if (GVR.VarSize != GV.VarSize) {
      std::cerr << "VarSize differs " << GVR.VarSize << " " << GV.VarSize
                << "\n";
      return -1;
    }

    if (std::memcmp(GVR.HostAddr.get(), GV.HostAddr.get(), 128) != 0) {
      std::cerr << "Memory differs between GV and GVR\n";
      return -1;
    }
    return 0;
  }();

  delete[] HData;
  HData = nullptr;

  EC = Vendor::DeviceErrorCheck(Vendor::DeviceFree(DData));
  if (EC)
    FATAL_ERROR("Could not release device memory\n");

  return Ret;
}
