//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#ifndef _COMMON_HPP
#define _COMMON_HPP

#pragma once

#include <chrono>

namespace common {

using Duration = std::chrono::duration<double>;

class MyTimer {
 public:
   MyTimer() : start(std::chrono::steady_clock::now()) {}

  Duration elapsed() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<Duration>(now - start);
  }

 private:
  std::chrono::steady_clock::time_point start;
};

};  // namespace common

#endif
