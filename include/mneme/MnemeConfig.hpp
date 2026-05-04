#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace mneme {

enum class LogLevel { Trace, Debug, Info, Warn, Error, Critical, Off };
enum class EpilogueSnapshotType { Bytes, Diff };

class Config {
public:
  static Config &get();

  static Config createFromEnvironment();

  const std::optional<std::string> KernelRegex;
  const uint64_t MaxRecordings;
  const uint64_t SkipRecordings;
  const std::optional<long> PageSizeGiB;
  const LogLevel MnemeLogLevel;
  const EpilogueSnapshotType EpilogueType;

  bool isRecordingEnabledForCurrentRank() const;

  std::filesystem::path getDataDirectory() const;

  uint64_t getPageSizeBytesOrDefault(long DefaultPageSizeGiB) const;

  std::optional<std::string> getLogDirectory() const;

private:
  const std::optional<std::string> MnemeDataDir;
  const std::optional<std::string> MnemeLogDir;
  const bool RecordingEnabledThisRank;

  Config();
};

} // namespace mneme
