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
  // Accepts the prologue and epilogue in either argument order; each role is
  // resolved by asking both states which one they are.
  auto *S1 = unwrap(v1);
  auto *S2 = unwrap(v2);

  auto *Prologue = S1->asPrologue() ? S1->asPrologue() : S2->asPrologue();
  auto *Epilogue = S1->asEpilogue() ? S1->asEpilogue() : S2->asEpilogue();
  if (!Prologue || !Epilogue)
    LOG_FATAL("CompareMemState expects one prologue and one epilogue");

  return Epilogue->matches(*Prologue);
}

API_EXPORT(void)
MnemePy_ResetMemState(MnemeDeviceMemStateRef MemState) {
  auto state = unwrap(MemState);
  state->reset();
}
}
