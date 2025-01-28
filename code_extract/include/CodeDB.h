#pragma once
#include "clang/Basic/LLVM.h"
#include "clang/Frontend/ASTUnit.h"

#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

namespace clang {
class Decl;
class ASTUnit;
} // namespace clang

/// @brief Stores uniquely identifying information for AST nodes of interest.
class ObjInfo {
  clang::ASTUnit const &unit;
  std::string const keyName;
  clang::Decl *decl;
  clang::Decl *def = nullptr;
  // Track the spec decl as we need to know the correct instantiation to be able
  // to infer param types and lambda uses.
  clang::Decl *specDecl = nullptr;
  bool defInSameTU = false;
  // If this decl is external to the project, store its source file's name for
  // include'ing later.
  std::string extSourceFile = "";

  clang::Decl *getDef() const { return def ? def : decl; }

public:
  ObjInfo(clang::ASTUnit const &astUnit, std::string keyName,
          clang::Decl *mainDecl, clang::Decl *defDecl = nullptr)
      : unit(astUnit), keyName(keyName), decl(mainDecl), def(defDecl),
        defInSameTU(defDecl) {}
  void addDefinitionDecl(clang::Decl *defDecl) { def = defDecl; }
  void addSpecializationDecl(clang::Decl *decl) { specDecl = decl; }

  clang::Decl *getDefiniton() { return getDef(); }
  clang::Decl const *getDefiniton() const { return getDef(); }

  clang::Decl *getSpecialization() { return specDecl; }

  bool isDefInSameTU() const { return defInSameTU; }

  std::string getKeyName() const { return keyName; }

  /// Get filename from which this specific decl is referenced.
  clang::StringRef getRefFile() const {
    return unit.getOriginalSourceFileName();
  }

  void addExtSourceFile(std::string file) { extSourceFile = file; }

  std::string getExtSourceFile() const { return extSourceFile; }
};

class CodeDB {
  // mangling to ObjInfo
  std::unordered_map<std::string, std::unique_ptr<ObjInfo>> db;
  // qualified names to keys which can be manglings or plain names
  std::unordered_map<std::string, std::unordered_set<std::string>> manglings;

  clang::Decl *getDef(std::string const &keyName) const {
    if (isRegistered(keyName))
      return db.at(keyName)->getDefiniton();
    else
      return nullptr;
  }

  ObjInfo *getObjInfo(std::string const &keyName) const {
    if (isRegistered(keyName))
      return db.at(keyName).get();
    else
      return nullptr;
  }

  void registerDeclImpl(clang::ASTUnit const &unit, std::string keyName,
                        clang::NamedDecl *decl, clang::NamedDecl *defDecl);

public:
  std::string const projPath;

  CodeDB(std::string const &projDir) : projPath(projDir) {}

  static std::string getKeyName(clang::NamedDecl const *decl);

  bool isRegistered(std::string const &keyName) const {
    return db.find(keyName) != db.end();
  }

  void registerDecl(clang::ASTUnit const &unit, clang::NamedDecl *decl,
                    clang::NamedDecl *defDecl = nullptr);

  void registerDecl(clang::ASTUnit const &unit, clang::FunctionDecl *decl,
                    clang::FunctionDecl *defDecl = nullptr);

  void registerDecl(clang::ASTUnit const &unit,
                    clang::FunctionTemplateDecl *decl,
                    clang::FunctionTemplateDecl *defDecl = nullptr);

  ObjInfo const *getObjInfoOrNull(std::string const &keyName) const {
    return getObjInfo(keyName);
  }

  ObjInfo *getObjInfoOrNull(std::string const &keyName) {
    return getObjInfo(keyName);
  }

  void getManglings(std::string const &name,
                    std::unordered_set<std::string> &mangle) const {
    if (manglings.find(name) == manglings.end())
      mangle = {};
    else
      mangle = manglings.at(name);
  }

  clang::Decl const *getDefinitionDecl(std::string const &keyName) const {
    return getDef(keyName);
  }

  clang::Decl *getDefinitionDecl(std::string const &keyName) {
    return getDef(keyName);
  }

  void addDefinitionDecl(std::string const &keyName, clang::Decl *defDecl) {
    if (isRegistered(keyName))
      db.at(keyName)->addDefinitionDecl(defDecl);
  }

  void addExtSource(std::string const &keyName,
                    std::string const &fileName) {
    if (isRegistered(keyName))
      db.at(keyName)->addExtSourceFile(fileName);
  }

  std::string getExtSource(std::string const &keyName) const {
    if (db.find(keyName) == db.end())
      return "";
    return db.at(keyName)->getExtSourceFile();
  }
};