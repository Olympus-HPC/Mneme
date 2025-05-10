#include "llvm/core.h"
#include <hip/hip_runtime_api.h>
#include <llvm/Support/CBindingWrapping.h>
#include <llvm/Support/MemoryBuffer.h>
#include <mneme/DeviceTraits.hpp>
#include <mneme/Utils.hpp>

using namespace mneme;
using namespace llvm;

#ifdef MNEME_ENABLE_HIP
#include "mneme/MnemeRecordHIP.hpp"
using MnemeRecorderDevice = MnemeRecorderHIP;
using DeviceVendorTraits = DeviceTraits<DeviceVendors::HIP>;
constexpr DeviceVendors Vendor = DeviceVendors::HIP;
#endif

extern "C" {

API_EXPORT(void *)
MnemePY_initializePageManager(uintptr_t Addr, uint64_t VASize) {
  void *VAddr = reinterpret_cast<void *>(Addr);
  auto PM = initializePageManager<DeviceVendorTraits>(VAddr, VASize);
  if (PM->getVAStart() != VAddr) {
    FATAL_ERROR("Could not allocate Device Pages\n Record got : " +
                util::pointerToHexString(VAddr) + " and replay got : " +
                util::pointerToHexString(PM->getVAStart()));
  }
  void *PMPtr = PM.release();
  return PMPtr;
}

API_EXPORT(void) MnemePY_DisposePageManager(void *PMPtr) {
  PageManager *PM = reinterpret_cast<PageManager *>(PMPtr);
  if (!PM) {
    FATAL_ERROR("Calling Dispose of page manager with a null pointer");
  }
  delete PM;
}
}
