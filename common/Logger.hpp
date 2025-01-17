#pragma once
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>

namespace mneme {

class Logger {
public:
  static std::ofstream &logs(const std::string &Name) {
    static Logger SingletonLogger{Name};
    return SingletonLogger.OutStream;
  }

  static std::ostream &warn() { return std::cerr; }

  static void flush(const std::string &Name) {
    auto &Logger = logs(Name);
    Logger.flush();
  }

private:
  const std::string LogDir = "mneme-logs";
  bool DirExists;
  std::error_code EC;
  std::ofstream OutStream;

  Logger(std::string Name) {
    DirExists = std::filesystem::create_directory(LogDir);
    if (false) {
      OutStream = std::ofstream{
          LogDir + "/" + Name + "." + std::to_string(getpid()) + ".log",
      };
    } else {
      OutStream.std::basic_ios<char>::rdbuf(std::cout.rdbuf());
    }
    if (!OutStream.good())
      throw std::runtime_error("Error opening file: " + EC.message());
  }

  void flush() { OutStream.flush(); }
};

} // namespace mneme
