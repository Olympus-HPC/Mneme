#include "mneme/MnemeEnv.hpp"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

namespace mneme {
namespace env_detail {

std::optional<int> parseInteger(std::string_view Text) {
  if (Text.empty())
    return std::nullopt;

  std::string NullTerminated(Text);
  errno = 0;
  char *End = nullptr;
  long Parsed = std::strtol(NullTerminated.c_str(), &End, 10);
  if (errno == ERANGE || End == NullTerminated.c_str() || *End != '\0' ||
      Parsed > INT_MAX || Parsed < INT_MIN)
    return std::nullopt;

  return static_cast<int>(Parsed);
}

void warnMalformedEnvironmentValue(const char *Description, const char *Name,
                                   std::string_view Value, const char *Suffix) {
  std::ostringstream Message;
  Message << "[mneme] Ignoring malformed " << Description << " " << Name << "='"
          << Value << "'";
  if (Suffix && Suffix[0] != '\0')
    Message << " " << Suffix;

  std::cerr << Message.str() << "\n";
}

} // namespace env_detail
} // namespace mneme
