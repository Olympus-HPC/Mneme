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
