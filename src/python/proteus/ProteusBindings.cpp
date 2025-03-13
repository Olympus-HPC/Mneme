#include <proteus/CoreLLVM.hpp>
#include <proteus/CoreLLVMDevice.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace proteus;

namespace proteus::python {

// Global variable to ensure LLVM is only initialized once
static bool llvm_initialized = false;

// Function to initialize LLVM targets (called automatically)
void init_llvm_once() {
  if (!llvm_initialized) {
    static proteus::InitLLVMTargets
        llvm_initializer; // Static ensures one-time init
    llvm_initialized = true;
  }
}

} // namespace proteus::python

PYBIND11_MODULE(proteus, m) {
  m.doc() = "Python bindings for Proteus JIT";

  // Expose function to initialize LLVM (but we’ll call it automatically)
  m.def("_init_llvm", &proteus::python::init_llvm_once,
        "Initialize LLVM targets");
}
