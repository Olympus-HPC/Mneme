#include "Logger.hpp"
#include "MnemeMemoryHIP.hpp"
#include "Utils.hpp"
#include <hip/hip_runtime.h>
#include <vector>
using namespace mneme;

using MemoryAllocationHandle_t = hipMemGenericAllocationHandle_t;

struct Blob {
  MemoryAllocationHandle_t MHandle;
  void *Addr;
  size_t Size;
  Blob() {};
  void release() {
    hipErrCheck(hipMemUnmap((void *)Addr, Size));
    hipErrCheck(hipMemRelease(MHandle));
    hipErrCheck(hipMemAddressFree((void *)Addr, Size));
  }
};

int main(int argc, char *argv[]) {
  size_t Size = std::atoi(argv[1]);
  std::vector<Blob> Blobs;
  for (int i = 0; i < 10; i++) {
    int DeviceID = 0;
    void *Addr;
    Blobs.emplace_back();
    auto &MBlob = Blobs.back();
    hipMemAllocationProp Prop = {};
    auto MinPageSize = MnemeMemoryBlobHIP::getMinPageSize(DeviceID);
    Prop.type = hipMemAllocationTypePinned;
    auto ActualSize = util::roundUp(Size, MinPageSize);
    MBlob.Addr =
        MnemeMemoryBlobHIP::getVirtualAddress(ActualSize, Addr, MinPageSize);
    Prop.location.type = hipMemLocationTypeDevice;
    Prop.location.id = 0;
    hipErrCheck(hipMemCreate(&MBlob.MHandle, ActualSize, &Prop, 0));
    hipErrCheck(hipMemMap((void *)MBlob.Addr, ActualSize, 0, MBlob.MHandle, 0));

    hipMemAccessDesc ADesc = {};
    ADesc.location.type = hipMemLocationTypeDevice;
    ADesc.location.id = DeviceID;
    ADesc.flags = hipMemAccessFlagsProtReadWrite;

    hipErrCheck(hipMemSetAccess(MBlob.Addr, ActualSize, &ADesc, 1));
    MBlob.Size = ActualSize;
    // Sets address
  }
  for (auto &B : Blobs) {
    DBG(Logger::logs("mneme") << "Got Addr " << std::hex << B.Addr << std::dec
                              << " with size " << B.Size << "\n");
    B.release();
  }
  Blobs.clear();
}
