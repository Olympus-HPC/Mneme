#include "mneme/MnemeLogger.hpp"

#ifdef MNEME_ENABLE_LOGGER
namespace mneme {
namespace MnemeLogger {
struct LoggerInitializer {
  LoggerInitializer() { MnemeLogger::initialize(); }
};

static LoggerInitializer initializer;
} // namespace MnemeLogger
} // namespace mneme
#endif
