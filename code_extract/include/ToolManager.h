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

public:
  ToolManager(std::string const &projectDirPath);

  /// @brief Finds the function declaration by name.
  /// @param fnName Name of the function declaration to find.
  /// @param mangledName If provided, mangled name of the function specialization to find.
  /// @return The function declaration if a function by that name exists in the
  /// tool's compilation database, null otherwise.
  ObjInfo *findFnDeclByName(std::string const &fnName,
                            std::string mangledName = "");

  /// @brief Pulls the function definition for the specified function name and
  /// all its dependencies into file <fn-name>.<fn-source-extn> and compiles it
  /// to <fn-name>.o.
  /// @param fnName Name of the function to get dependencies for.
  /// @param mangledFnName If the function is a template function, also specify
  /// which function spec to pull.
  void getStandaloneFnContext(std::string const &fnName,
                              std::string mangledFnName = "");
};
