#pragma once

#include <optional>
#include <string_view>

namespace mneme {
namespace env_detail {

std::optional<int> parseNonNegativeInteger(std::string_view Text);
void warnMalformedEnvironmentValue(
    const char *Description, const char *Name, std::string_view Value,
    const char *Suffix = "", std::optional<int> DetectedRank = std::nullopt);

} // namespace env_detail
} // namespace mneme
