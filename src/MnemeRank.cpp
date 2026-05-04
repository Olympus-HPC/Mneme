#include "mneme/MnemeRank.hpp"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <unistd.h>

namespace mneme {
namespace {

void warnMalformedEnv(const char *Name, const char *Value) {
  static std::once_flag WarnOnce;
  std::call_once(WarnOnce, [Name, Value]() {
    std::cerr << "[mneme] Ignoring malformed rank environment variable " << Name
              << "='" << Value << "'\n";
  });
}

std::optional<int> parseIntEnv(const char *Name, bool AllowZero) {
  const char *Value = std::getenv(Name);
  if (!Value)
    return std::nullopt;

  if (Value[0] == '\0') {
    warnMalformedEnv(Name, Value);
    return std::nullopt;
  }

  errno = 0;
  char *End = nullptr;
  long Parsed = std::strtol(Value, &End, 10);
  if (errno == ERANGE || End == Value || *End != '\0' || Parsed > INT_MAX ||
      Parsed < (AllowZero ? 0 : 1)) {
    warnMalformedEnv(Name, Value);
    return std::nullopt;
  }

  return static_cast<int>(Parsed);
}

std::optional<int> firstValidEnv(const char *const *Names, bool AllowZero) {
  for (const char *Name = *Names; Name; Name = *++Names) {
    auto Parsed = parseIntEnv(Name, AllowZero);
    if (Parsed)
      return Parsed;
  }
  return std::nullopt;
}

} // namespace

std::optional<int> detectDistributedRank() {
  static const char *const RankVars[] = {
      "FLUX_TASK_RANK",       "OMPI_COMM_WORLD_RANK", "PMI_RANK",
      "MPI_RANK",             "SLURM_PROCID",         "JSM_NAMESPACE_RANK",
      "PMIX_RANK",            "PBS_TASKNUM",          nullptr};
  return firstValidEnv(RankVars, true);
}

std::optional<int> detectDistributedSize() {
  static const char *const SizeVars[] = {
      "FLUX_JOB_SIZE", "FLUX_NTASKS",         "OMPI_COMM_WORLD_SIZE",
      "PMI_SIZE",      "SLURM_NTASKS",        "PMIX_SIZE",
      nullptr};
  return firstValidEnv(SizeVars, false);
}

std::string getRankIdString() {
  if (auto Rank = detectDistributedRank())
    return std::to_string(*Rank);
  return std::to_string(getpid());
}

} // namespace mneme
