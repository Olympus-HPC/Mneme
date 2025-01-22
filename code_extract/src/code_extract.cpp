#include <filesystem>
#include <iostream>

#include "ToolManager.h"

int main(int argc, const char **argv) {
  if (argc < 3 || argc > 4) {
    std::cerr
        << "Usage: code-extract path/to/dir function-name [mangled-name]\n"
        << "Please ensure the path points to the root folder of your project!\n"
        << "If requesting to pull a function template, please specify the mangled name of the specialization to pull.\n";
    return 1;
  }
  std::string dirPath = argv[1];
  std::string fnName = argv[2]; // Possibly needs to be fully qualified...
  std::string mangledFnName = "";
  if (argc == 4)  mangledFnName = argv[3];

  if (!std::filesystem::is_directory(dirPath)) {
    std::cerr << "Error: Directory '" << dirPath << "' not found.\n";
    return 1;
  }

  // set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE), need the compilation database to
  // recall all compile commands and source files.
  // Or use bear -- make in project directory
  ToolManager tm(dirPath);

  // For not this function does all the work but we should probably make a
  // cleaner interface...
  tm.getStandaloneFnContext(fnName, mangledFnName);
}