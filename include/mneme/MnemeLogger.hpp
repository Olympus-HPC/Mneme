#pragma once

#ifdef MNEME_ENABLE_LOGGING

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

namespace MnemeLogger {
inline std::shared_ptr<spdlog::logger> logger;

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
    return spdlog::level::info; // Default level

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

inline void initialize() {
  std::string logDir = getLogDirectory();
  if (!logDir.empty()) {
    std::string logFile = logDir + "/" + getLogFileName();
    logger = spdlog::basic_logger_mt("file_logger", logFile);
    logger->set_pattern("[mneme] [%^%l%$] %v");
  } else {
    logger = spdlog::stdout_color_mt("console_logger");
    logger->set_pattern("[\033[34mmneme\033[0m] [%^%l%$] %v");
  }
  logger->set_level(getLogLevelFromEnv());
}
} // namespace MnemeLogger
} // namespace mneme

// Logging macros
#define LOG_DEBUG(...) mneme::MnemeLogger::logger->debug(__VA_ARGS__)
#define LOG_INFO(...) mneme::MnemeLogger::logger->info(__VA_ARGS__)
#define LOG_WARN(...) mneme::MnemeLogger::logger->warn(__VA_ARGS__)
#define LOG_CRITICAL(...) mneme::MnemeLogger::logger->critical(__VA_ARGS__)

#else // Logging disabled

#define LOG_DEBUG(...) ((void)0)
#define LOG_INFO(...) ((void)0)
#define LOG_WARN(...) ((void)0)
#define LOG_CRITICAL(...) ((void)0)

#endif // ENABLE_LOGGING
