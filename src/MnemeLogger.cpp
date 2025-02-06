#include "mneme/MnemeLogger.hpp"

#ifdef MNEME_ENABLE_LOGGING
namespace mneme {
namespace MnemeLogger {
struct LoggerInitializer {
  LoggerInitializer() { MnemeLogger::initialize(); }
};

static LoggerInitializer initializer;
} // namespace MnemeLogger
} // namespace mneme
#endif
