#include "llvm/core.h"
#include <llvm/Support/CBindingWrapping.h>
#include <llvm/Support/MemoryBuffer.h>

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemePageManager.hpp"
#include "mneme/MnemeUtils.hpp"

using namespace mneme;
using namespace llvm;

#ifdef MNEME_ENABLE_HIP
using DeviceVendorTraits = DeviceTraits<DeviceVendors::HIP>;
constexpr DeviceVendors Vendor = DeviceVendors::HIP;
#elif defined(MNEME_ENABLE_CUDA)
using DeviceVendorTraits = DeviceTraits<DeviceVendors::CUDA>;
constexpr DeviceVendors Vendor = DeviceVendors::CUDA;
#endif

extern "C" {

API_EXPORT(void *)
MnemePY_initializePageManager(int DeviceID, uintptr_t Addr, uint64_t VASize) {
  LOG_DEBUG("Initializing page manager for Device {}", DeviceID);
  void *VAddr = reinterpret_cast<void *>(Addr);
  auto PM = initializePageManager<Vendor>(DeviceID, VAddr, VASize);
  if (PM->getVAStart() != VAddr) {
    LOG_WARN("Could not reserve the exact recorded VA base; replay will use "
             "reserved base {} instead of recorded base {}",
             PM->getVAStart(), VAddr);
  }
  void *PMPtr = PM.release();
  return PMPtr;
}

API_EXPORT(uintptr_t) MnemePY_getPageManagerVAStart(void *PMPtr) {
  auto *PM = reinterpret_cast<PageManager<Vendor> *>(PMPtr);
  if (!PM)
    LOG_FATAL("Calling getVAStart of page manager with a null pointer");
  return reinterpret_cast<uintptr_t>(PM->getVAStart());
}

API_EXPORT(void) MnemePY_DisposePageManager(void *PMPtr) {
  PageManager<Vendor> *PM = reinterpret_cast<PageManager<Vendor> *>(PMPtr);
  if (!PM) {
    LOG_FATAL("Calling Dispose of page manager with a null pointer");
  }
  delete PM;
}
}
