#include "mneme/MnemeConfig.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>

using namespace mneme;

namespace {

constexpr uint64_t GiB = 1024L * 1024L * 1024L;

void expect(bool Condition, const std::string &Message) {
  if (!Condition) {
    std::cerr << Message << "\n";
    std::exit(1);
  }
}

void clearMnemeEnv() {
  unsetenv("MNEME_RR_KERNELS");
  unsetenv("MNEME_DATA_DIR");
  unsetenv("MNEME_MAX_RECORDINGS");
  unsetenv("MNEME_SKIP_RECORDINGS");
  unsetenv("MNEME_PAGE_SIZE");
  unsetenv("MNEME_LOG_LEVEL");
  unsetenv("MNEME_LOG_DIR");
  unsetenv("MNEME_EPILOGUE_TYPE");
}

std::filesystem::path makeTempDir() {
  auto Path = std::filesystem::temp_directory_path() /
              ("mneme-config-test-" + std::to_string(getpid()));
  std::filesystem::create_directories(Path);
  return Path;
}

} // namespace

int main() {
  clearMnemeEnv();
  {
    auto Conf = Config::createFromEnvironment();
    expect(!Conf.KernelRegex, "MNEME_RR_KERNELS should default to unset");
    expect(Conf.getDataDirectory() == std::filesystem::current_path(),
           "MNEME_DATA_DIR should default to current path");
    expect(Conf.MaxRecordings == 4, "MNEME_MAX_RECORDINGS should default to 4");
    expect(Conf.SkipRecordings == 0,
           "MNEME_SKIP_RECORDINGS should default to 0");
    expect(!Conf.PageSizeGiB, "MNEME_PAGE_SIZE should default to unset");
    expect(Conf.getPageSizeBytesOrDefault(64) == 64 * GiB,
           "HIP page size fallback should be 64 GiB");
    expect(Conf.getPageSizeBytesOrDefault(2) == 2 * GiB,
           "CUDA page size fallback should be 2 GiB");
    expect(Conf.MnemeLogLevel == LogLevel::Critical,
           "MNEME_LOG_LEVEL should default to critical");
    expect(!Conf.getLogDirectory(), "MNEME_LOG_DIR should default to unset");
    expect(Conf.EpilogueType == EpilogueSnapshotType::Bytes,
           "MNEME_EPILOGUE_TYPE should default to bytes");
  }

  auto TempDir = makeTempDir();
  setenv("MNEME_RR_KERNELS", "_two", 1);
  setenv("MNEME_DATA_DIR", TempDir.c_str(), 1);
  setenv("MNEME_MAX_RECORDINGS", "8", 1);
  setenv("MNEME_SKIP_RECORDINGS", "2", 1);
  setenv("MNEME_PAGE_SIZE", "3", 1);
  setenv("MNEME_LOG_LEVEL", "debug", 1);
  setenv("MNEME_LOG_DIR", TempDir.c_str(), 1);
  setenv("MNEME_EPILOGUE_TYPE", "diff", 1);
  {
    auto Conf = Config::createFromEnvironment();
    expect(Conf.KernelRegex && *Conf.KernelRegex == "_two",
           "MNEME_RR_KERNELS should preserve the regex string");
    expect(Conf.getDataDirectory() == std::filesystem::absolute(TempDir),
           "MNEME_DATA_DIR should use the configured directory");
    expect(Conf.MaxRecordings == 8,
           "MNEME_MAX_RECORDINGS should use the configured value");
    expect(Conf.SkipRecordings == 2,
           "MNEME_SKIP_RECORDINGS should use the configured value");
    expect(Conf.PageSizeGiB && *Conf.PageSizeGiB == 3,
           "MNEME_PAGE_SIZE should parse as GiB");
    expect(Conf.getPageSizeBytesOrDefault(64) == 3 * GiB,
           "MNEME_PAGE_SIZE should override backend defaults");
    expect(Conf.MnemeLogLevel == LogLevel::Debug,
           "MNEME_LOG_LEVEL should map debug");
    expect(Conf.getLogDirectory() && *Conf.getLogDirectory() == TempDir,
           "MNEME_LOG_DIR should use the configured directory");
    expect(Conf.EpilogueType == EpilogueSnapshotType::Diff,
           "MNEME_EPILOGUE_TYPE should map diff");
  }

  setenv("MNEME_MAX_RECORDINGS", "12abc", 1);
  setenv("MNEME_SKIP_RECORDINGS", "6abc", 1);
  setenv("MNEME_PAGE_SIZE", "5abc", 1);
  setenv("MNEME_LOG_LEVEL", "verbose", 1);
  {
    auto Conf = Config::createFromEnvironment();
    expect(Conf.MaxRecordings == 12,
           "MNEME_MAX_RECORDINGS should keep atoi-style parsing");
    expect(Conf.SkipRecordings == 6,
           "MNEME_SKIP_RECORDINGS should keep atoi-style parsing");
    expect(Conf.PageSizeGiB && *Conf.PageSizeGiB == 5,
           "MNEME_PAGE_SIZE should keep atol-style parsing");
    expect(Conf.MnemeLogLevel == LogLevel::Info,
           "invalid MNEME_LOG_LEVEL should fall back to info");
  }

  setenv("MNEME_MAX_RECORDINGS", "abc", 1);
  setenv("MNEME_SKIP_RECORDINGS", "abc", 1);
  setenv("MNEME_PAGE_SIZE", "abc", 1);
  setenv("MNEME_EPILOGUE_TYPE", "bytes", 1);
  {
    auto Conf = Config::createFromEnvironment();
    expect(Conf.MaxRecordings == 0,
           "invalid MNEME_MAX_RECORDINGS should parse to 0");
    expect(Conf.SkipRecordings == 0,
           "invalid MNEME_SKIP_RECORDINGS should parse to 0");
    expect(Conf.PageSizeGiB && *Conf.PageSizeGiB == 0,
           "invalid MNEME_PAGE_SIZE should parse to 0");
    expect(Conf.EpilogueType == EpilogueSnapshotType::Bytes,
           "MNEME_EPILOGUE_TYPE should map bytes");
  }

  setenv("MNEME_EPILOGUE_TYPE", "delta", 1);
  {
    bool Threw = false;
    try {
      auto Conf = Config::createFromEnvironment();
      (void)Conf;
    } catch (const std::runtime_error &) {
      Threw = true;
    }
    expect(Threw, "invalid MNEME_EPILOGUE_TYPE should throw");
  }

  clearMnemeEnv();
  std::filesystem::remove_all(TempDir);
  return 0;
}
