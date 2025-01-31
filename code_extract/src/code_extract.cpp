#include <filesystem>
#include <iostream>

#include <llvm/Support/CommandLine.h>

#include "ToolManager.h"

namespace cli_opt {

const llvm::cl::opt<std::string>
    dirPath(llvm::cl::Positional, llvm::cl::Required,
            llvm::cl::desc("<Path to target compile_commands.json>"));
const llvm::cl::list<std::string>
    fnName(llvm::cl::Positional, llvm::cl::Required, llvm::cl::OneOrMore,
           llvm::cl::desc("<Function name(s) to pull>"));
const llvm::cl::opt<std::string> mangledFnName(
    "m", llvm::cl::desc(
             "Resolve ambiguous pull with mangled function name (optional)"));
const llvm::cl::opt<bool>
    emitRR("emitRR",
           llvm::cl::desc("Emit Record Replay specific function hooks"),
           llvm::cl::init(false));
const llvm::cl::opt<bool> emitAllDecls(
    "emitAllDecls",
    llvm::cl::desc(
        "Emit all function declarations for the given function name"),
    llvm::cl::init(false));
const llvm::cl::opt<bool> pullExternal(
    "pullExternal",
    llvm::cl::desc(
        "Include external decls in created database. Only use this option if you wish to pull external function calls."),
    llvm::cl::init(false));
} // namespace cli_opt

int main(int argc, const char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "code-extract - Extract function source code\n");
  if (cli_opt::dirPath.empty() || cli_opt::fnName.empty()) {
    std::cerr << "Error: Both directory path and function name are required.\n"
              << "Usage: code-extract <directory-path> <function-name(s)> [-m "
                 "<mangled-name>] [--emitRR]\n";
    return 1;
  }

  if (!cli_opt::emitAllDecls && cli_opt::fnName.size() > 1) {
    std::cerr << "Pulling more than one function at once is only supported "
                 "with the emitAllDecls option!";
    return 1;
  }

  if (!std::filesystem::is_directory(cli_opt::dirPath.data())) {
    std::cerr << "Error: Directory '" << cli_opt::dirPath << "' not found.\n";
    return 1;
  }

  // set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE), need the compilation database to
  // recall all compile commands and source files.
  // Or use bear -- make in project directory
  ToolManager tm(cli_opt::dirPath, cli_opt::emitRR, cli_opt::pullExternal);

  if (!cli_opt::emitAllDecls) {
    tm.getStandaloneFnContext(cli_opt::fnName[0], cli_opt::fnName[0],
                              cli_opt::mangledFnName);
  } else {
    for (auto& file : cli_opt::fnName)
      tm.getAllDeclarations(file);
  }
}