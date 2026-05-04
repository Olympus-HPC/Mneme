#include "mneme/MnemeRank.hpp"

#include "mneme/MnemeConfig.hpp"

#include <cstdlib>

namespace mneme {
namespace {

std::optional<int> firstSetRankEnv(const char *const *Names) {
  for (const char *Name = *Names; Name; Name = *++Names) {
    const char *Value = std::getenv(Name);
    if (!Value)
      continue;

    auto Parsed = config_detail::parseInteger(Value);
    if (!Parsed || *Parsed < 0) {
      config_detail::warnMalformedEnvironmentValue("rank environment variable",
                                                   Name, Value);
      return std::nullopt;
    }
    return Parsed;
  }
  return std::nullopt;
}

} // namespace

std::optional<int> detectDistributedRank() {
  static const char *const RankVars[] = {
      "FLUX_TASK_RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK",
      "MPI_RANK",       "SLURM_PROCID",         "JSM_NAMESPACE_RANK",
      "PMIX_RANK",      "PBS_TASKNUM",          nullptr};
  return firstSetRankEnv(RankVars);
}

} // namespace mneme
