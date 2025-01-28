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
#include <unordered_set>

namespace ct = clang::tooling;

namespace helper {
std::string
getCompilationFlags(std::vector<std::string> const &cli,
                    std::unordered_set<std::string> const &blacklistedFlags) {
  std::string flags;
  bool prev = false;
  for (auto &command : cli) {
    if ((prev || command[0] == '-') &&
                 blacklistedFlags.find(command) == blacklistedFlags.end()) {
      flags += command + " ";
      prev = !prev;
    }
  }
  return flags;
}
} // namespace helper

ToolManager::ToolManager(std::string const &dirPath, bool emitRR) : emitRR(emitRR) {
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
  db.reset(new CodeDB(std::filesystem::canonical(dirPath)));
  // Build code database
  for (auto &ast : asts) {
    CodeExtractVisitor vis(*db.get(), *ast, dirPath);
    // Should move this in a run function or ctor
    vis.TraverseAST(ast->getASTContext());
  }
}

ObjInfo *ToolManager::findFnDeclByName(std::string const &fnName,
                                       std::string mangledName) {
  auto obj = db->getObjInfoOrNull(fnName);

  if (!obj) {
    std::cerr << "Could not find function(s) named " << fnName
              << " within specified project!" << std::endl;
    exit(1);
  }
  if (!obj->getDefiniton()) {
    std::cerr << "Could not find body for function(s) named " << fnName
              << " within specified project! Make sure it is not an externally "
                 "defined function!"
              << std::endl;
    exit(1);
  }

  auto fnDecl = static_cast<clang::FunctionDecl *>(obj->getDefiniton());
  clang::ASTNameGenerator astNameGen(fnDecl->getASTContext());
  if (auto tmpDecl = fnDecl->getDescribedFunctionTemplate()) {
    if (mangledName.empty()) {
      std::cerr << "Extraction of function template " << fnName
                << " requested without specifying specialization to pull. "
                << "Please specify the mangled name of the specialization you "
                   "wish to extract. Or present a non empty string to view the "
                   "list of reachable specializations."
                << std::endl;
      exit(1);
    }
    clang::Decl *specDecl = nullptr;
    std::string candidates;
    for (auto it = tmpDecl->spec_begin(); it != tmpDecl->spec_end(); it++) {
      auto itName = astNameGen.getName(*it);
      if (mangledName == itName) {
        specDecl = *it;
        obj->addSpecializationDecl(specDecl);
        break;
      }
      candidates += "Mangled name: " + itName + "\n";
    }
    if (!specDecl) {
      std::cerr << "Could not find specialization for " << fnName
                << " with mangled name " << mangledName << "!" << std::endl;
      if (candidates.empty()) {
        std::cerr << "No candidates to report! Are you sure this function "
                     "template is being instantiated in the project scope?"
                  << std::endl;
      } else {
        std::cerr << "\nSpecializations found for " << fnName << ":\n"
                  << candidates << "\n"
                  << std::endl;
      }
      exit(1);
    }
  }

  return obj;
}

void ToolManager::getStandaloneFnContext(std::string const &fnName,
                                         std::string const &mangledFnName) {
  primaryFn = findFnDeclByName(fnName, mangledFnName);
  auto fnSrcFile = primaryFn->getRefFile();
  bool isCuda = fnSrcFile.substr(fnSrcFile.size() - 2, 2) == "cu";

  std::string filename = std::regex_replace(fnName, std::regex("[^\\w]"), "_");
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
      cli[0] + " -o " + objname + " " + filename + " " +
      helper::getCompilationFlags(cli, {"-c", "-o", "--driver-mode=g++", "--"});
  command += " " + defaultOpts;
  std::cout << "Compiling " << fnName << " with command:\n"
            << command << std::endl;
  if (system(command.c_str()) != 0) {
    std::cerr << "Compilation failed!" << std::endl;
  } else {
    std::cout << "Compilation successful!" << std::endl;
  }
}