#include "llvm/core.h"
#include <memory>
#include <mneme/MnemePython.hpp>
#include <mneme/MnemeUtils.hpp>
#include <string>

using namespace mneme;
using namespace mneme::python;
using namespace llvm;
// namespace

extern "C" {
API_EXPORT(MnemeDeviceMemStateRef)
MnemePy_initializeMemState(const char *KernelName, const char *fn,
                           const char *BasePrologueFn, bool isPrologue) {
  std::string BaseSnapshotName =
      BasePrologueFn == nullptr ? "" : std::string(BasePrologueFn);
  std::unique_ptr<DeviceMemState> state =
      isPrologue
          ? makeReplayPrologueState<Vendor>(KernelName, fn)
          : makeReplayEpilogueState<Vendor>(KernelName, fn, BaseSnapshotName);
  return wrap(state.release());
}

API_EXPORT(void) MnemePy_DisposeMemState(MnemeDeviceMemStateRef MemState) {
  if (MemState == nullptr)
    return;
  auto state = unwrap(MemState);
  delete state;
}

API_EXPORT(void) MnemePy_LoadMemState(MnemeDeviceMemStateRef MemState) {
  auto state = unwrap(MemState);
  state->load();
}

API_EXPORT(bool)
MnemePy_CompareMemState(MnemeDeviceMemStateRef v1, MnemeDeviceMemStateRef v2) {
  // Python calls this as prologue.__eq__(epilogue): v1 is the prologue and v2
  // the epilogue.
  auto *Prologue = unwrap(v1)->asPrologue();
  auto *Epilogue = unwrap(v2)->asEpilogue();
  if (!Prologue || !Epilogue)
    LOG_FATAL("CompareMemState expects (prologue, epilogue)");
  return Epilogue->matches(*Prologue);
}

API_EXPORT(void)
MnemePy_ResetMemState(MnemeDeviceMemStateRef MemState) {
  auto state = unwrap(MemState);
  state->reset();
}
}
