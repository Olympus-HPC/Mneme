// Copyright (c) 2021-2024. LLNL-CODE-2000766 and Mneme Contributors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mneme/MnemePageManager.hpp"
#include "mneme/MnemeUtils.hpp"
#include <cstdint>

using namespace mneme;

ContiguousAddrBlock::ContiguousAddrBlock(uintptr_t start, uint64_t sz)
    : PageAddr(start), Size(sz) {}

// Comparison operators for sorting blocks by address and size
bool ContiguousAddrBlock::operator<(const ContiguousAddrBlock &other) const {
  return PageAddr < other.PageAddr;
}
