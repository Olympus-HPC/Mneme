#pragma once
#include <memory>
#include <vector>

#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Tooling.h"

#include "CodeDB.h"

namespace clang {
class ASTUnit;
class FunctionDecl;
} // namespace clang

/// @brief Exposes API to pull a single function by name and manages other tool
/// aspects.
class ToolManager {
  std::unique_ptr<clang::tooling::CompilationDatabase> compDb;
  std::unique_ptr<clang::tooling::ClangTool> tool;
  ObjInfo *primaryFn;
  std::vector<std::unique_ptr<clang::ASTUnit>> asts;
  std::unique_ptr<CodeDB> db;
  bool emitRR;

public:
  ToolManager(std::string const &projectDirPath, bool emitRR);

  /// @brief Finds the function declaration by name.
  /// @param fnName Name of the function declaration to find.
  /// @param mangledName If provided, mangled name of the function
  /// specialization to find.
  /// @return The function declaration if a function by that name exists in the
  /// tool's compilation database, null otherwise.
  ObjInfo *findFnDeclByName(std::string const &fnName,
                            std::string mangledName = "");

  /// @brief Pulls the function definition for the specified function name and
  /// all its dependencies into file <fn-name>.<fn-source-extn> and compiles it
  /// to <fn-name>.o. If outFileName is specified, that name is used instead.
  /// @param fnName Name of the function to get dependencies for.
  /// @param outFileName Name of the file to which the standalone code is written to.
  /// @param mangledFnName If the function has multiple decls, specify
  /// which decl to pull.
  void getStandaloneFnContext(std::string const &fnName, std::string const& outFileName = "",
                              std::string mangledFnName = "");

  // Similar to getStandaloneFnContext but this pulls all possible declarations
  // with the name fnName into individual standalone files named <fnName><declaration-no>.<fn-source-ext>.
  void getAllDeclarations(std::string const &fnName);
};
