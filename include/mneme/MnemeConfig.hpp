#pragma once

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>

namespace mneme {

enum class LogLevel { Trace, Debug, Info, Warn, Error, Critical, Off };
enum class EpilogueSnapshotType { Bytes, Diff };

namespace config_detail {

inline std::optional<std::string> getEnvOrDefaultString(const char *VarName) {
  const char *EnvValue = std::getenv(VarName);
  if (!EnvValue)
    return std::nullopt;

  return std::string(EnvValue);
}

inline uint64_t getEnvOrDefaultIntLenient(const char *VarName,
                                          uint64_t Default) {
  const char *EnvValue = std::getenv(VarName);
  return EnvValue ? static_cast<uint64_t>(std::atoi(EnvValue)) : Default;
}

inline std::optional<long> getEnvOrDefaultPageSizeGiB(const char *VarName) {
  const char *EnvValue = std::getenv(VarName);
  if (!EnvValue)
    return std::nullopt;

  return std::atol(EnvValue);
}

inline uint64_t pageSizeGiBToBytes(long PageSizeGiB) {
  return static_cast<uint64_t>(PageSizeGiB * 1024L * 1024L * 1024L);
}

inline LogLevel getEnvOrDefaultLogLevel(const char *VarName, LogLevel Default) {
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

inline EpilogueSnapshotType getEnvOrDefaultEpilogueSnapshotType(
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

} // namespace config_detail

class Config {
public:
  static Config &get() {
    static Config Conf;
    return Conf;
  }

  static Config createFromEnvironment() { return Config(); }

  const std::optional<std::string> KernelRegex;
  const uint64_t MaxRecordings;
  const uint64_t SkipRecordings;
  const std::optional<long> PageSizeGiB;
  const LogLevel MnemeLogLevel;
  const EpilogueSnapshotType EpilogueType;

  std::filesystem::path getDataDirectory() const {
    auto Dir = MnemeDataDir.value_or(std::filesystem::current_path().string());
    std::filesystem::path Path(Dir);
    if (!std::filesystem::is_directory(Path)) {
      throw std::runtime_error("Path :" + Path.string() + " does not exist.\n");
    }
    return std::filesystem::absolute(Path);
  }

  uint64_t getPageSizeBytesOrDefault(long DefaultPageSizeGiB) const {
    return config_detail::pageSizeGiBToBytes(
        PageSizeGiB.value_or(DefaultPageSizeGiB));
  }

  std::optional<std::string> getLogDirectory() const {
    if (MnemeLogDir && !std::filesystem::exists(*MnemeLogDir)) {
      throw std::runtime_error("'MNEME_LOG_DIR' directory does not exist\n");
    }
    return MnemeLogDir;
  }

private:
  const std::optional<std::string> MnemeDataDir;
  const std::optional<std::string> MnemeLogDir;

  Config()
      : KernelRegex(config_detail::getEnvOrDefaultString("MNEME_RR_KERNELS")),
        MaxRecordings(config_detail::getEnvOrDefaultIntLenient(
            "MNEME_MAX_RECORDINGS", 4)),
        SkipRecordings(config_detail::getEnvOrDefaultIntLenient(
            "MNEME_SKIP_RECORDINGS", 0)),
        PageSizeGiB(
            config_detail::getEnvOrDefaultPageSizeGiB("MNEME_PAGE_SIZE")),
        MnemeLogLevel(config_detail::getEnvOrDefaultLogLevel(
            "MNEME_LOG_LEVEL", LogLevel::Critical)),
        EpilogueType(config_detail::getEnvOrDefaultEpilogueSnapshotType(
            "MNEME_EPILOGUE_TYPE", EpilogueSnapshotType::Bytes)),
        MnemeDataDir(config_detail::getEnvOrDefaultString("MNEME_DATA_DIR")),
        MnemeLogDir(config_detail::getEnvOrDefaultString("MNEME_LOG_DIR")) {}
};

} // namespace mneme
