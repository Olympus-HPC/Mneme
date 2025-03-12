#include "mneme/MnemeLogger.hpp"

#ifdef MNEME_ENABLE_LOGGER
namespace mneme {
namespace MnemeLogger {
struct LoggerInitializer {
private:
  spdlog::logger logger;
  LoggerInitializer() {
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

public:
  static spdlog::logger &getLogger() {
    static LoggerInitializer;
    return LoggerInitializer;
  }
}
};

} // namespace MnemeLogger
} // namespace mneme
#endif
