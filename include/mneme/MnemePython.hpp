#include <mneme/DeviceTraits.hpp>
#include <mneme/MnemeReplay.hpp>

#ifdef MNEME_ENABLE_HIP
using DeviceVendorTraits = mneme::DeviceTraits<mneme::DeviceVendors::HIP>;
constexpr mneme::DeviceVendors Vendor = mneme::DeviceVendors::HIP;
using DeviceMemState = mneme::ReplayMemState<Vendor>;
#elif defined(MNEME_ENABLE_CUDA)
using DeviceVendorTraits = mneme::DeviceTraits<mneme::DeviceVendors::CUDA>;
constexpr mneme::DeviceVendors Vendor = mneme::DeviceVendors::CUDA;
using DeviceMemState = mneme::ReplayMemState<Vendor>;
#endif

typedef DeviceMemState *MnemeDeviceMemStateRef;

namespace mneme {
namespace python {
static MnemeDeviceMemStateRef wrap(DeviceMemState *W) {
  return reinterpret_cast<MnemeDeviceMemStateRef>(W);
}
static DeviceMemState *unwrap(MnemeDeviceMemStateRef WR) {
  return reinterpret_cast<DeviceMemState *>(WR);
}

} // namespace python
} // namespace mneme
