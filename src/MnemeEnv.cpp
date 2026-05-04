#include "mneme/MnemeEnv.hpp"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <set>
#include <sstream>
#include <string>

namespace mneme {
namespace env_detail {

std::optional<int> parseNonNegativeInteger(std::string_view Text) {
  if (Text.empty())
    return std::nullopt;

  std::string NullTerminated(Text);
  errno = 0;
  char *End = nullptr;
  long Parsed = std::strtol(NullTerminated.c_str(), &End, 10);
  if (errno == ERANGE || End == NullTerminated.c_str() || *End != '\0' ||
      Parsed > INT_MAX || Parsed < 0)
    return std::nullopt;

  return static_cast<int>(Parsed);
}

void warnMalformedEnvironmentValue(const char *Description, const char *Name,
                                   std::string_view Value, const char *Suffix,
                                   std::optional<int> DetectedRank) {
  if (DetectedRank && *DetectedRank != 0)
    return;

  std::ostringstream Message;
  Message << "[mneme] Ignoring malformed " << Description << " " << Name << "='"
          << Value << "'";
  if (Suffix && Suffix[0] != '\0')
    Message << " " << Suffix;

  static std::mutex Mutex;
  static std::set<std::string> Emitted;
  auto Text = Message.str();
  std::lock_guard<std::mutex> Lock(Mutex);
  if (Emitted.insert(Text).second)
    std::cerr << Text << "\n";
}

} // namespace env_detail
} // namespace mneme
