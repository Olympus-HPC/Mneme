#pragma once

#include <optional>
#include <string_view>

namespace mneme {
namespace env_detail {

std::optional<int> parseInteger(std::string_view Text);
void warnMalformedEnvironmentValue(const char *Description, const char *Name,
                                   std::string_view Value,
                                   const char *Suffix = "");

} // namespace env_detail
} // namespace mneme
