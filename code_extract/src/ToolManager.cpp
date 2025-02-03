#include "ToolManager.h"
#include "CodeDB.h"
#include "Visitor.h"

#include <clang/AST/Mangle.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <string>
#include <unordered_set>

namespace ct = clang::tooling;

namespace helper {
std::string
getCompilationFlags(std::vector<std::string> const &cli,
                    std::unordered_set<std::string> const &blacklistedFlags) {
  std::string flags;
  bool canBeArg = true;
  for (auto &command : cli) {
    bool isFlag = command[0] == '-';
    if ((canBeArg || isFlag) &&
        blacklistedFlags.find(command) == blacklistedFlags.end()) {
      flags += command + " ";
      canBeArg = isFlag;
    } else {
      canBeArg = false;
    }
  }
  return flags;
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

  auto sourceFiles = compDb->getAllFiles();
  tool = std::make_unique<ct::ClangTool>(*compDb, sourceFiles);

  // Build code database
  tool->buildASTs(asts);
  db.reset(new CodeDB(std::filesystem::canonical(dirPath), includeExternals));
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

  if (!obj->getDefiniton()) {
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
  auto& ctx = primaryFn->getDefiniton()->getASTContext();
  auto fnSrcFile = ctx.getSourceManager().getFilename(primaryFn->getDefiniton()->getLocation());
  bool isCuda = ctx.getLangOpts().CUDA;

  std::string filename;
  if (!outFileName.empty())
    filename = outFileName;
  else
    filename = fnName;
  filename = std::regex_replace(filename, std::regex("[^\\w]"), "_");
  std::string objname = filename + ".o";
  filename += (isCuda ? ".cu" : ".cpp");

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

  // Then compile
  // For now only get one compilation command
  auto cli = compDb->getCompileCommands(fnSrcFile)[0].CommandLine;
  // Remove warning for use of uninitialized vars, eventually make this optional
  std::string defaultOpts = "-Wno-uninitialized";
  command =
      helper::getCompilationFlags(cli, {"-c", "-o", "--driver-mode=g++", "--"});
  command += " " + defaultOpts;
  command += " -o " + objname + " " + filename;
  std::cout << "Compiling " << fnName << " with command:\n"
            << command << std::endl;
  if (system(command.c_str()) != 0) {
    std::cerr << "Compilation failed!" << std::endl;
  } else {
    std::cout << "Compilation successful!" << std::endl;
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