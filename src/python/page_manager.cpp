// Copyright (c) 2021-2024. LLNL-CODE-2000766 and Mneme Contributors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
#endif

extern "C" {

API_EXPORT(void *)
MnemePY_initializePageManager(int DeviceID, uintptr_t Addr, uint64_t VASize) {
  LOG_DEBUG("Initializing page manager for Device {}", DeviceID);
  void *VAddr = reinterpret_cast<void *>(Addr);
  auto PM = initializePageManager<Vendor>(DeviceID, VAddr, VASize);
  if (PM->getVAStart() != VAddr) {
    LOG_FATAL("Could not allocate Device Pages\n Record got : " +
              util::pointerToHexString(VAddr) + " and replay got : " +
              util::pointerToHexString(PM->getVAStart()));
  }
  void *PMPtr = PM.release();
  return PMPtr;
}

API_EXPORT(void) MnemePY_DisposePageManager(void *PMPtr) {
  PageManager<Vendor> *PM = reinterpret_cast<PageManager<Vendor> *>(PMPtr);
  if (!PM) {
    LOG_FATAL("Calling Dispose of page manager with a null pointer");
  }
  delete PM;
}
}
