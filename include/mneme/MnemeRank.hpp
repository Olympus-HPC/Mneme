#pragma once

#include <optional>
#include <string>

namespace mneme {

std::optional<int> detectDistributedRank();
std::optional<int> detectDistributedSize();
std::string getRankIdString();

} // namespace mneme
