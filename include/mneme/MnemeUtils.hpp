// Copyright (c) 2021-2024. LLNL-CODE-2000766 and Mneme Contributors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

#if MNEME_ENABLE_DEBUG
#define MNEME_DBG(x) x;
#else
#define MNEME_DBG(x)
#endif

namespace mneme {
namespace util {
template <typename Ty> Ty roundUp(Ty Size, Ty Divider) {
  return (Size + Divider - 1) & ~(Divider - 1);
}
template <typename Ty> Ty extractScalar(const char *&Buffer) {
  Ty Value = *reinterpret_cast<const Ty *>(Buffer);
  Buffer += sizeof(Ty);
  return Value;
}
template <typename Ty> Ty extractScalar(char *&Buffer) {
  Ty Value = *reinterpret_cast<const Ty *>(Buffer);
  Buffer += sizeof(Ty);
  return Value;
}


template <typename T> std::string pointerToHexString(T *ptr) {
  std::ostringstream oss;
  oss << "0x" << std::hex << std::setw(sizeof(void *) * 2) << std::setfill('0')
      << reinterpret_cast<std::uintptr_t>(ptr);
  return oss.str();
}

template <typename T> T *hexStringToPointer(const std::string &HEXStr) {
  uint64_t Addr = std::stoul(HEXStr, nullptr, 16);
  // Cast the integer to void*
  return reinterpret_cast<T *>(Addr);
}

} // namespace util
} // namespace mneme
