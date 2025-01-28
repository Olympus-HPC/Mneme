#include <filesystem>
#include <iostream>

#include <llvm/Support/CommandLine.h>

#include "ToolManager.h"

const llvm::cl::opt<std::string> dirPath(llvm::cl::Positional,
                                         llvm::cl::Required,
                                         llvm::cl::desc("<directory path>"));
const llvm::cl::opt<std::string> fnName(llvm::cl::Positional,
                                        llvm::cl::Required,
                                        llvm::cl::desc("<function name>"));
const llvm::cl::opt<std::string>
    mangledFnName("m", llvm::cl::desc("Mangled function name (optional)"),
                  llvm::cl::value_desc("mangled-name"));
const llvm::cl::opt<bool>
    emitRR("emitRR",
           llvm::cl::desc("Emit Record Replay specific function hooks"),
           llvm::cl::init(false));

int main(int argc, const char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "code-extract - Extract function code\n");
  if (dirPath.empty() || fnName.empty()) {
    std::cerr << "Error: Both directory path and function name are required.\n"
              << "Usage: code-extract <directory-path> <function-name> [-m "
                 "<mangled-name>] [--emitRR]\n";
    return 1;
  }

  if (!std::filesystem::is_directory(dirPath.data())) {
    std::cerr << "Error: Directory '" << dirPath << "' not found.\n";
    return 1;
  }

  // set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE), need the compilation database to
  // recall all compile commands and source files.
  // Or use bear -- make in project directory
  ToolManager tm(dirPath, emitRR);

  // For not this function does all the work but we should probably make a
  // cleaner interface...
  tm.getStandaloneFnContext(fnName, mangledFnName);
}