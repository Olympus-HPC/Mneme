#pragma once

#include <cstddef>
#include <cstdint>

#include "mneme/DeviceTraits.hpp"

namespace mneme {

struct DiffResetScatterTask {
  uint8_t *Dst = nullptr;
  const uint8_t *Src = nullptr;
  size_t Size = 0;
};

template <DeviceVendors VendorTypes>
typename DeviceTraits<VendorTypes>::DeviceError_t launchDiffResetScatterKernel(
    const DiffResetScatterTask *Tasks, size_t NumTasks,
    typename DeviceTraits<VendorTypes>::DeviceStream_t Stream);

} // namespace mneme
