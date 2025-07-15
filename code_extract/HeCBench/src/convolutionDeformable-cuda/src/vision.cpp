
#include "dcn_v2.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dcn_v2_forward", &dcn_v2_forward, "dcn_v2_forward");
  m.def("dcn_v2_backward", &dcn_v2_backward, "dcn_v2_backward");
}
