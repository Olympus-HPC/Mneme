#pragma once

#ifdef MNEME_ENABLE_LOGGER

#include "mneme/MnemeConfig.hpp"
#include "spdlog/spdlog.h"
#include <climits>
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

inline spdlog::level::level_enum toSpdLogLevel(LogLevel Level) {
  switch (Level) {
  case LogLevel::Trace:
    return spdlog::level::trace;
  case LogLevel::Debug:
    return spdlog::level::debug;
  case LogLevel::Info:
    return spdlog::level::info;
  case LogLevel::Warn:
    return spdlog::level::warn;
  case LogLevel::Error:
    return spdlog::level::err;
  case LogLevel::Critical:
    return spdlog::level::critical;
  case LogLevel::Off:
    return spdlog::level::off;
  }
  return spdlog::level::info;
}

inline std::string getLogDirectory() {
  try {
    auto Dir = Config::get().getLogDirectory();
    return Dir.value_or("");
  } catch (const std::runtime_error &Error) {
    std::cerr << Error.what();
    exit(-1);
  }
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
    _logger->set_level(toSpdLogLevel(Config::get().MnemeLogLevel));
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
  do {                                                                         \
  std::cout << "Error in file" << std::string(__FILE__) << ":" << __LINE__     \
            << "\n";                                                           \
  abort();                                                                     \
  } while (0)
#endif // ENABLE_LOGGING
