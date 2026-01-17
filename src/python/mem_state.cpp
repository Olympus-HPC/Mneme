// Copyright (c) 2021-2024. LLNL-CODE-2000766 and Mneme Contributors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/core.h"
#include <mneme/MnemePython.hpp>
#include <mneme/MnemeUtils.hpp>

using namespace mneme;
using namespace mneme::python;
using namespace llvm;
// namespace

extern "C" {
API_EXPORT(MnemeDeviceMemStateRef)
MnemePy_initializeMemState(const char *KernelName, const char *fn,
                           bool isPrologue) {
  DeviceMemState::InstanceType type =
      isPrologue ? DeviceMemState::InstanceType::Prologue
                 : DeviceMemState::InstanceType::Epilogue;
  DeviceMemState *state = new DeviceMemState(KernelName, fn, type);
  return wrap(state);
}

API_EXPORT(void) MnemePy_DisposeMemState(MnemeDeviceMemStateRef MemState) {
  auto state = unwrap(MemState);
  state->release();
}

API_EXPORT(void) MnemePy_LoadMemState(MnemeDeviceMemStateRef MemState) {
  auto state = unwrap(MemState);
  state->load();
}

API_EXPORT(bool)
MnemePy_CompareMemState(MnemeDeviceMemStateRef v1, MnemeDeviceMemStateRef v2) {
  auto state1 = unwrap(v1);
  auto state2 = unwrap(v2);
  auto res = (*state1 == *state2);
  return res;
}

API_EXPORT(void)
MnemePy_ResetMemState(MnemeDeviceMemStateRef MemState) {
  auto state = unwrap(MemState);
  state->reset();
}
}
