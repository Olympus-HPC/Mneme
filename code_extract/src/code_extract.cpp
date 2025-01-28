#include <filesystem>
#include <iostream>

#include <llvm/Support/CommandLine.h>

#include "ToolManager.h"

namespace cli_opt {

const llvm::cl::opt<std::string>
    dirPath(llvm::cl::Positional, llvm::cl::Required,
            llvm::cl::desc("<Path to target compile_commands.json>"));
const llvm::cl::opt<std::string>
    fnName(llvm::cl::Positional, llvm::cl::Required,
           llvm::cl::desc("<Function name to pull>"));
const llvm::cl::opt<std::string>
    mangledFnName("m", llvm::cl::desc("Resolve ambiguous pull with mangled function name (optional)"));
const llvm::cl::opt<bool>
    emitRR("emitRR",
           llvm::cl::desc("Emit Record Replay specific function hooks"),
           llvm::cl::init(false));
const llvm::cl::opt<bool> emitAllDecls(
    "emitAllDecls",
    llvm::cl::desc(
        "Emit all function declarations for the given function name"),
    llvm::cl::init(false));
}

int main(int argc, const char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "code-extract - Extract function source code\n");
  if (cli_opt::dirPath.empty() || cli_opt::fnName.empty()) {
    std::cerr << "Error: Both directory path and function name are required.\n"
              << "Usage: code-extract <directory-path> <function-name> [-m "
                 "<mangled-name>] [--emitRR]\n";
    return 1;
  }

  if (!std::filesystem::is_directory(cli_opt::dirPath.data())) {
    std::cerr << "Error: Directory '" << cli_opt::dirPath << "' not found.\n";
    return 1;
  }

  // set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE), need the compilation database to
  // recall all compile commands and source files.
  // Or use bear -- make in project directory
  ToolManager tm(cli_opt::dirPath, cli_opt::emitRR);

  if (!cli_opt::emitAllDecls)
    tm.getStandaloneFnContext(cli_opt::fnName, cli_opt::fnName, cli_opt::mangledFnName);
  else
    tm.getAllDeclarations(cli_opt::fnName);
}