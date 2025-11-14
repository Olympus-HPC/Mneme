#pragma once

#ifdef MNEME_ENABLE_LOGGER

#include "spdlog/spdlog.h"
#include <cstdlib> // getenv
#include <filesystem>
#include <iostream>
#include <memory>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <string>
#include <unistd.h>

namespace mneme {

namespace {

inline std::string getLogFileName() {
  char hostname[HOST_NAME_MAX];
  if (gethostname(hostname, HOST_NAME_MAX) != 0) {
    std::cerr << "Could not read host name\n";
    exit(-1);
  }

  std::string id;
  if (const char *rid = std::getenv("SLURM_PROCID")) {
    id = std::string(rid);
  } else if (const char *jsm = std::getenv("JSM_NAMESPACE_RANK")) {
    id = std::string(jsm);
  } else if (const char *pmi = std::getenv("PMIX_RANK")) {
    id = std::stoi(pmi);
  } else {
    id = std::to_string(getpid());
  }
  id = "mneme-" + std::string(hostname) + "-" + id + ".log";
  return id;
}

// Get log level from the environment
inline spdlog::level::level_enum getLogLevelFromEnv() {
  const char *env_p = std::getenv("MNEME_LOG_LEVEL");
  if (!env_p)
    return spdlog::level::off; // Default level

  static const std::unordered_map<std::string, spdlog::level::level_enum>
      logLevels = {{"trace", spdlog::level::trace},
                   {"debug", spdlog::level::debug},
                   {"info", spdlog::level::info},
                   {"warn", spdlog::level::warn},
                   {"error", spdlog::level::err},
                   {"critical", spdlog::level::critical},
                   {"off", spdlog::level::off}};

  std::string levelStr(env_p);
  auto it = logLevels.find(levelStr);
  return (it != logLevels.end()) ? it->second : spdlog::level::info;
}

// Get log directory from the environment
inline std::string getLogDirectory() {
  const char *dir = std::getenv("MNEME_LOG_DIR");
  if (dir && !std::filesystem::exists(dir)) {
    std::cerr << "'MNEME_LOG_DIR' directory does not exist\n";
    exit(-1);
  }
  return (dir != nullptr) ? std::string(dir) : "";
}

} // namespace

struct MnemeLogger {
private:
  std::shared_ptr<spdlog::logger> _logger;
  MnemeLogger() {
    std::string logDir = getLogDirectory();
    if (!logDir.empty()) {
      std::string logFile = logDir + "/" + getLogFileName();
      _logger = spdlog::basic_logger_mt("file_logger", logFile);
      _logger->set_pattern("[mneme] [%^%l%$] %v");
    } else {
      _logger = spdlog::stdout_color_mt("console_logger");
      _logger->set_pattern("[\033[34mmneme\033[0m] [%^%l%$] %v");
    }
    _logger->set_level(getLogLevelFromEnv());
  }

public:
  static spdlog::logger &getLogger() {
    static MnemeLogger logger;
    return *logger._logger;
  }
};

} // namespace mneme

// Logging macros
#define LOG_DEBUG(...) mneme::MnemeLogger::getLogger().debug(__VA_ARGS__)
#define LOG_INFO(...) mneme::MnemeLogger::getLogger().info(__VA_ARGS__)
#define LOG_WARN(...) mneme::MnemeLogger::getLogger().warn(__VA_ARGS__)
#define LOG_CRITICAL(...) mneme::MnemeLogger::getLogger().critical(__VA_ARGS__)
#define LOG_FATAL(...)                                                         \
  do {                                                                         \
    mneme::MnemeLogger::getLogger().critical(__VA_ARGS__);                     \
    mneme::MnemeLogger::getLogger().critical(                                  \
        "Error occured in file {} at line {}", __FILE__, __LINE__);            \
    abort();                                                                   \
  } while (0)

#else // Logging disabled
#include <iostream>
#define LOG_DEBUG(...) ((void)0)
#define LOG_INFO(...) ((void)0)
#define LOG_WARN(...) ((void)0)
#define LOG_CRITICAL(...) ((void)0)
#define LOG_FATAL(...)                                                         \
  std::cerr << "Error in file" << std::string(__FILE__) << ":" << __LINE__     \
            << "\n";                                                           \
  abort();
#endif // ENABLE_LOGGING
