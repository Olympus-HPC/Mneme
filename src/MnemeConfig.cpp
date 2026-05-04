#include "mneme/MnemeConfig.hpp"

#include "mneme/MnemeEnv.hpp"
#include "mneme/MnemeRank.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <string>

namespace mneme {
namespace {

std::optional<std::string> getEnvOrDefaultString(const char *VarName) {
  const char *EnvValue = std::getenv(VarName);
  if (!EnvValue)
    return std::nullopt;

  return std::string(EnvValue);
}

uint64_t getEnvOrDefaultIntLenient(const char *VarName, uint64_t Default) {
  const char *EnvValue = std::getenv(VarName);
  return EnvValue ? static_cast<uint64_t>(std::atoi(EnvValue)) : Default;
}

std::optional<long> getEnvOrDefaultPageSizeGiB(const char *VarName) {
  const char *EnvValue = std::getenv(VarName);
  if (!EnvValue)
    return std::nullopt;

  return std::atol(EnvValue);
}

uint64_t pageSizeGiBToBytes(long PageSizeGiB) {
  return static_cast<uint64_t>(PageSizeGiB * 1024L * 1024L * 1024L);
}

LogLevel getEnvOrDefaultLogLevel(const char *VarName, LogLevel Default) {
  auto EnvValue = getEnvOrDefaultString(VarName);
  if (!EnvValue)
    return Default;

  if (*EnvValue == "trace")
    return LogLevel::Trace;
  if (*EnvValue == "debug")
    return LogLevel::Debug;
  if (*EnvValue == "info")
    return LogLevel::Info;
  if (*EnvValue == "warn")
    return LogLevel::Warn;
  if (*EnvValue == "error")
    return LogLevel::Error;
  if (*EnvValue == "critical")
    return LogLevel::Critical;
  if (*EnvValue == "off")
    return LogLevel::Off;

  return LogLevel::Info;
}

EpilogueSnapshotType getEnvOrDefaultEpilogueSnapshotType(
    const char *VarName, EpilogueSnapshotType Default) {
  auto EnvValue = getEnvOrDefaultString(VarName);
  if (!EnvValue)
    return Default;

  if (*EnvValue == "bytes")
    return EpilogueSnapshotType::Bytes;
  if (*EnvValue == "diff")
    return EpilogueSnapshotType::Diff;

  throw std::runtime_error("Invalid MNEME_EPILOGUE_TYPE value '" + *EnvValue +
                           "'. Expected 'bytes' or 'diff'.");
}

bool defaultRecordingPolicy(const std::optional<int> &DetectedRank) {
  if (!DetectedRank)
    return true;
  return *DetectedRank == 0;
}

std::optional<std::set<int>>
parseRecordRanksList(const char *VarName, const std::string &Value,
                     const std::optional<int> &DetectedRank) {
  std::set<int> Parsed;
  size_t Start = 0;
  while (Start <= Value.size()) {
    size_t Comma = Value.find(',', Start);
    std::string Token = Value.substr(
        Start, Comma == std::string::npos ? std::string::npos : Comma - Start);

    auto ParsedRank = env_detail::parseNonNegativeInteger(Token);
    if (!ParsedRank) {
      env_detail::warnMalformedEnvironmentValue(
          "environment variable", VarName, Value,
          "(using default recording policy)", DetectedRank);
      return std::nullopt;
    }

    Parsed.insert(*ParsedRank);
    if (Comma == std::string::npos)
      break;
    Start = Comma + 1;
  }

  return Parsed;
}

bool computeRecordingEnabledForCurrentRank() {
  auto DetectedRank = detectDistributedRank();
  const char *EnvValue = std::getenv("MNEME_RECORD_RANKS");
  if (!EnvValue || EnvValue[0] == '\0')
    return defaultRecordingPolicy(DetectedRank);

  std::string Value(EnvValue);
  std::string Lower = Value;
  std::transform(Lower.begin(), Lower.end(), Lower.begin(),
                 [](unsigned char C) { return std::tolower(C); });
  if (Lower == "all")
    return true;

  auto ParsedRanks =
      parseRecordRanksList("MNEME_RECORD_RANKS", Value, DetectedRank);
  if (!ParsedRanks)
    return defaultRecordingPolicy(DetectedRank);

  return ParsedRanks->count(DetectedRank.value_or(0)) > 0;
}

} // namespace

Config &Config::get() {
  static Config Conf;
  return Conf;
}

Config Config::createFromEnvironment() { return Config(); }

bool Config::isRecordingEnabledForCurrentRank() const {
  return RecordingEnabledThisRank;
}

std::filesystem::path Config::getDataDirectory() const {
  auto Dir = MnemeDataDir.value_or(std::filesystem::current_path().string());
  std::filesystem::path Path(Dir);
  if (!std::filesystem::is_directory(Path))
    throw std::runtime_error("Path :" + Path.string() + " does not exist.\n");
  return std::filesystem::absolute(Path);
}

uint64_t Config::getPageSizeBytesOrDefault(long DefaultPageSizeGiB) const {
  return pageSizeGiBToBytes(PageSizeGiB.value_or(DefaultPageSizeGiB));
}

std::optional<std::string> Config::getLogDirectory() const {
  if (MnemeLogDir && !std::filesystem::exists(*MnemeLogDir))
    throw std::runtime_error("'MNEME_LOG_DIR' directory does not exist\n");
  return MnemeLogDir;
}

Config::Config()
    : KernelRegex(getEnvOrDefaultString("MNEME_RR_KERNELS")),
      MaxRecordings(getEnvOrDefaultIntLenient("MNEME_MAX_RECORDINGS", 4)),
      SkipRecordings(getEnvOrDefaultIntLenient("MNEME_SKIP_RECORDINGS", 0)),
      PageSizeGiB(getEnvOrDefaultPageSizeGiB("MNEME_PAGE_SIZE")),
      MnemeLogLevel(
          getEnvOrDefaultLogLevel("MNEME_LOG_LEVEL", LogLevel::Critical)),
      EpilogueType(getEnvOrDefaultEpilogueSnapshotType(
          "MNEME_EPILOGUE_TYPE", EpilogueSnapshotType::Bytes)),
      MnemeDataDir(getEnvOrDefaultString("MNEME_DATA_DIR")),
      MnemeLogDir(getEnvOrDefaultString("MNEME_LOG_DIR")),
      RecordingEnabledThisRank(computeRecordingEnabledForCurrentRank()) {}

} // namespace mneme
