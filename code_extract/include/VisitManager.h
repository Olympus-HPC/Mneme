#pragma once

#include <cstddef>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "CodeDB.h"

namespace clang {
class NamedDecl;
class Decl;
class Expr;
class FunctionDecl;
class TagDecl;
class VarDecl;
class Stmt;
class ASTContext;
class Type;
} // namespace clang

class MatchVisitor;

/// @brief Manages nodes to visit and decls to add to the extracted body.
class VisitManager {
  CodeDB const &db;
  ObjInfo &primaryFn;
  std::queue<clang::Stmt *> toVisitNodes;
  std::vector<clang::NamedDecl const *> varDeclRefs;
  std::vector<clang::NamedDecl const *> declRefs;
  std::vector<clang::NamedDecl const *> tagDecls;
  std::vector<clang::NamedDecl const *> typedefDeclsUnexpanded;
  std::vector<clang::NamedDecl const *> typedefDecls;
  // Unordered maintenance of namespace aliases, if the aliases depend on each
  // other then we give up :)
  std::unordered_set<clang::NamespaceAliasDecl const *> namespaceAliases;
  // All named decls whose defs are a part of a different decl (i.e. typedefs)
  // and hence should not be emitted separately, currently only for tag types.
  std::unordered_set<clang::NamedDecl const *> hasIndirectDef;
  // Keeps tracks of primary template decls so we don't emit them multiple
  // times.
  std::unordered_set<clang::NamedDecl const *> tempDecls;
  std::unordered_set<clang::NamedDecl const *> fwdDecls;
  std::unordered_map<std::string, ObjInfo const *> visitedNodes;
  std::unordered_set<std::size_t> includes;
  std::vector<std::pair<std::string, std::string>> params_expr;
  std::vector<std::pair<std::string, clang::ValueDecl const *>> params_decl;

  /// @brief Fills params from the given vals into the params map with string id
  /// as prefix + param_position.
  template <typename T>
  void fillParams(std::string const &prefix, T *begin, T *end,
                  MatchVisitor &mv);

public:
  VisitManager(ObjInfo &pf, CodeDB const &cdb) : db(cdb), primaryFn(pf) {}

  /// @brief Check if a named decl is visited or not.
  /// @param keyName Key name of the decl to check.
  bool isVisited(std::string const &keyName);

  /// @brief Gets the the visited objInfo by Key name.
  ObjInfo const *getVisitedObj(std::string const &keyName);

  /// @brief Marks a named decl visited.
  /// @param name Key name of the decl to mark.
  /// @param objInfo Object info associated with decl to mark.
  void markVisited(std::string const &keyName, ObjInfo const *objInfo);

  /// @brief Marks a decl to not be independently emitted (currently only for
  /// tag types.)
  void markIndirectDef(clang::TagDecl const *decl);

  /// @brief Adds the correct alias namespace decl for a given namespace
  void addIfNamespaceAliased(std::string const &namesp);

  /// @brief Adds a declaration to be emitted to code later.
  void registerDecl(clang::NamedDecl const *decl);

  /// @brief Adds a var decl to be emitted to code later.
  void registerDecl(clang::VarDecl const *decl);

  /// @brief Adds a function declaration to be emitted later. DO NOT call this
  /// for the primary function!
  /// @param decl Function declaration.
  void registerDecl(clang::FunctionDecl const *decl);

  /// @brief Adds the declaration of an enum/struct/class to be emitted
  /// later.
  /// @param decl Decl of the record type.
  void registerDecl(clang::RecordDecl const *decl);

  /// @brief Adds the declaration of an typedefs to be emitted
  /// later.
  /// @param decl Decl of the typedef type.
  void registerDecl(clang::TypedefNameDecl const *decl);

  /// @brief Adds the the includePath to the files to include. Note: the input
  /// path must be project internal.
  void registerInclude(std::string const &includePath);

  /// @brief Register a function call as a forward decl.
  void fwdDeclFuncDecl(clang::FunctionDecl const *decl);

  /// @brief Default instantiates all function parameters for the given
  /// functionDecl and returns a vector of decl strings for each param.
  /// @param fnObj The function ObjInfo to instantiate params for.
  void getParamInstantiationsAsString(ObjInfo *fnObj,
                                      std::vector<std::string> &params);

  /// @brief Registers all variable parameters that will be used to build the
  /// standalone function call for fnObj. These params may be params to fnObj or
  /// to function call/lambda expression params.
  void registerParameterPrologue(MatchVisitor &mv, ObjInfo *fnObj);

  /// @brief Emits the standalone code file containing the searched function and
  /// all other dependencies into output.
  /// @param output String to dump the full standalone code into.
  /// @param configString For CUDA global functions specifying grid, block and
  /// thread size.
  void emitStandaloneFile(std::string &output, bool emitRR,
                          std::string const &configString = "");

  /// @brief Given a function declaration, pulls all dependencies necessary to
  /// make the function compile standalone.
  /// @param fnDecl The function declaration to pull dependencies for.
  void pullPrimaryFnContext();
};