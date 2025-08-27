#include "ToolManager.h"
#include "CodeDB.h"
#include "Visitor.h"
#include "clang/Basic/FileEntry.h"

#include <clang/AST/Mangle.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <ostream>
#include <regex>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace ct = clang::tooling;

namespace helper {
std::tuple<std::string, std::string>
getCompilationFlags(std::vector<std::string> const &cli,
                    std::unordered_set<std::string> const &blacklistedFlags) {
  std::string flags;
  bool canBeArg = true;
  for (int i = 1; i < cli.size(); i++) {
    auto command = cli[i];
    bool isFlag = command[0] == '-';
    if ((canBeArg || isFlag) &&
        blacklistedFlags.find(command) == blacklistedFlags.end()) {
      flags += command + " ";
      canBeArg = isFlag;
    } else {
      // If current flag is arg, remove the previous flag too
      if (canBeArg && !isFlag)
        flags = flags.substr(0, flags.find_last_of(' ', flags.size() - 2));
      canBeArg = false;
    }
  }
  return {cli[0], flags};
}
} // namespace helper

ToolManager::ToolManager(std::string const &dirPath, bool emitRR,
                         bool includeExternals)
    : emitRR(emitRR) {
  // Setup our tool
  std::string errorMsg;
  compDb = ct::CompilationDatabase::autoDetectFromDirectory(dirPath, errorMsg);
  if (!errorMsg.empty()) {
    std::cerr << errorMsg << std::endl;
    exit(1);
  }

  if (compDb->getAllCompileCommands().empty()) {
    std::cerr << "Compile commands in " << dirPath
              << " is empty! No functions will be found even if they exist."
              << std::endl;
    exit(1);
  }
  auto canDirPath = std::filesystem::canonical(dirPath);

  auto sourceFiles = compDb->getAllFiles();
  tool = std::make_unique<ct::ClangTool>(*compDb, sourceFiles);

  /// FIXME: eventually we should collect includes all over
  auto command = compDb->getAllCompileCommands()[0].CommandLine;
  std::unordered_set<std::string> includePaths;
  for (auto const &flag : command) {
    auto idx = flag.find("-I");
    if (idx == std::string::npos)
      continue;
    auto filePath = flag.substr(idx + 2);
    if (filePath.find(canDirPath) != std::string::npos) {
      includePaths.insert(filePath);
    }
  }

  // Build code database
  tool->buildASTs(asts);
  db.reset(new CodeDB(canDirPath, includePaths, includeExternals));
  // Build code database
  for (auto &ast : asts) {
    CodeExtractVisitor vis(*db.get(), *ast, dirPath);
    // Should move this in a run function or ctor
    vis.TraverseAST(ast->getASTContext());
  }
}

ObjInfo *ToolManager::findFnDeclByName(std::string const &fnName,
                                       std::string mangledName) {
  std::unordered_set<std::string> manglings;
  db->getManglings(fnName, manglings);
  ObjInfo *obj = nullptr;

  if (manglings.empty()) {
    std::cerr << "Could not find function(s) named " << fnName
              << " within specified project!" << std::endl;
    if (!db->includeExternals)
      std::cerr << "If this is an external function call, please use the "
                   "--pullExternal flag!";
    exit(1);
  }

  if (manglings.size() > 1 && mangledName.empty()) {
    std::cerr
        << "Extraction of function " << fnName
        << " based only on its source name is ambiguous! Either the function "
           "is a function template or is overloaded. "
        << "Please specify the mangled name of the specific declaration you "
           "wish to extract. Or present a non empty string to view the "
           "list of reachable declarations with the same source name."
        << std::endl;
    exit(1);
  } else if (manglings.size() == 1) {
    obj = db->getObjInfoOrNull(*manglings.begin());
  } else {
    if (manglings.find(mangledName) == manglings.end()) {
      std::cerr << "Could not find declaration for " << fnName
                << " with mangled name " << mangledName << "!" << std::endl;
      std::cerr << "\nSpecializations found for " << fnName << ":\n";
      for (auto name : manglings)
        std::cerr << name << std::endl;
      exit(1);
    } else {
      obj = db->getObjInfoOrNull(mangledName);
    }
  }

  if (!obj->getDefinition()) {
    std::cerr << "Could not find body for function(s) named " << fnName
              << " within specified project! Make sure it is not an externally "
                 "defined function!"
              << std::endl;
    exit(1);
  }

  return obj;
}

void ToolManager::getStandaloneFnContext(std::string const &fnName,
                                         std::string const &outFileName,
                                         std::string mangledFnName) {
  primaryFn = findFnDeclByName(fnName, mangledFnName);
  auto &ctx = primaryFn->getDefinition()->getASTContext();
  auto fnSrcFile = ctx.getSourceManager().getFilename(
      primaryFn->getDefinition()->getLocation());
  bool isHip = ctx.getLangOpts().HIP;
  bool isCuda = ctx.getLangOpts().CUDA && !isHip;
  bool isGPU = isCuda || isHip;

  std::string basename;
  if (!outFileName.empty())
    basename = outFileName;
  else
    basename = fnName;
  basename = std::regex_replace(basename, std::regex("[^\\w]"), "_");
  std::string objname = basename + ".o";
  std::string filename = basename + (isGPU ? ".cu" : ".cpp");

  VisitManager mv(*primaryFn, *db.get());

  mv.pullPrimaryFnContext();
  std::string code;
  mv.emitStandaloneFile(code, emitRR);

  // Compile pulled code
  // First dump everything to a file
  std::ofstream outFile(filename);

  if (!outFile) {
    std::cerr << "Error creating file " << filename << std::endl;
    return;
  }

  outFile << code;
  outFile.close();

  // Then clang-format
  std::string command = "clang-format -i " + filename;
  if (system(command.c_str()) != 0)
    std::cerr << "Formatting failed!" << std::endl;

  // For now only get one compilation command
  auto cli = compDb->getCompileCommands(fnSrcFile)[0].CommandLine;
  // Remove warning for use of uninitialized vars, eventually make this optional
  std::string defaultOpts = "-Wno-uninitialized";
  std::unordered_set<std::string> blacklistFlags = {"-triple",
                                                    "-filetype",
                                                    "-main-file-name",
                                                    "-target-cpu",
                                                    "-mrelocation-model",
                                                    "-cc1as",
                                                    "-o",
                                                    "--driver-mode=g++",
                                                    "--"};
  if (isGPU)
    blacklistFlags.insert("c++-header");
  auto [compiler, flags] = helper::getCompilationFlags(cli, blacklistFlags);
  // If its cuda, we will do 2 phase compilation + linking
  if (isHip)
    flags += " -x hip ";

  // Then remove unused includes
  command = "clang-tidy --quiet --checks='-*,misc-include-cleaner' -fix " +
            filename + " -- " + flags + " > /dev/null";
  if (system(command.c_str()) != 0)
    std::cerr << "Removing unused headers failed!" << std::endl;

  // Then compile
  command = compiler + " " + flags + " " + defaultOpts + " -o " + objname +
            " " + filename;
  std::cout << "Compiling " << fnName << " with command:\n"
            << command << std::endl;
  if (system(command.c_str()) != 0) {
    std::cerr << "Compilation failed!" << std::endl;
  } else {
    std::cout << "Compilation successful!" << std::endl;
    // For now, only link against basic libs for cuda, ideally we want to pull
    // all link flags from CC
    if (isGPU) {
      std::string linkCommand;
      if (isHip)
        linkCommand = "clang++ --hip-link " + objname + " -o " + basename;
      else 
        linkCommand =
            "clang++ " + objname +
            " -Wl,-rpath,${CUDA_LIBS} -L${CUDA_LIBS} -lcuda -lcudadevrt "
            "-lcudart_static -lrt -lpthread -ldl -o " +
            basename;
      if (system(linkCommand.c_str()) != 0) {
        if (isCuda && !std::getenv("CUDA_LIBS"))
          std::cout << "Point CUDA_LIBS to your driver "
                       "libraries, otherwise "
                       "linking WILL fail.";
        std::cout << "Linking failed!" << std::endl;
      } else
        std::cout << "Linking successful!" << std::endl;
    }
  }
}

void ToolManager::getAllDeclarations(std::string const &fnName) {
  std::unordered_set<std::string> manglings;
  db->getManglings(fnName, manglings);

  int cnt = 0;
  for (auto mangledName : manglings) {
    std::string filename =
        std::regex_replace(fnName, std::regex("[^\\w]"), "_");
    getStandaloneFnContext(fnName, fnName + std::to_string(cnt++), mangledName);
  }
}