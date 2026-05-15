#pragma once

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>

namespace mneme {

namespace rank_detail {

inline std::optional<int> parseInteger(std::string_view Text) {
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

inline void warnMalformedRankEnvironmentValue(const char *Name,
                                              std::string_view Value) {
  std::cerr << "[mneme] Ignoring malformed rank environment variable " << Name
            << "='" << Value << "'\n";
}

inline std::optional<int> firstSetRankEnv(const char *const *Names) {
  for (const char *Name = *Names; Name; Name = *++Names) {
    const char *Value = std::getenv(Name);
    if (!Value)
      continue;

    auto Parsed = parseInteger(Value);
    if (!Parsed || *Parsed < 0) {
      warnMalformedRankEnvironmentValue(Name, Value);
      continue;
    }
    return Parsed;
  }
  return std::nullopt;
}

} // namespace rank_detail

inline std::optional<int> detectDistributedRank() {
  static const char *const RankVars[] = {
      "FLUX_TASK_RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK",
      "MPI_RANK",       "SLURM_PROCID",         "JSM_NAMESPACE_RANK",
      "PMIX_RANK",      "PBS_TASKNUM",          nullptr};
  return rank_detail::firstSetRankEnv(RankVars);
}

} // namespace mneme
