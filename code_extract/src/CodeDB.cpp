#include "CodeDB.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Mangle.h"

#include <cassert>

std::string CodeDB::getKeyName(clang::NamedDecl const *decl) {
  auto enumDecl = llvm::dyn_cast<clang::EnumDecl>(decl);
  if (enumDecl && !enumDecl->getIdentifier()) {
    return "enum " + enumDecl->enumerator_begin()->getQualifiedNameAsString();
  } else if (llvm::dyn_cast<clang::TypeDecl>(decl)) {
    std::string prefix;
    if (decl->getKind() == clang::Decl::Kind::Typedef)
      prefix = "typedef ";
    return prefix + decl->getQualifiedNameAsString();
  } else
    return clang::ASTNameGenerator(decl->getASTContext()).getName(decl);
}

void CodeDB::registerDeclImpl(clang::ASTUnit const &unit, std::string keyName,
                              clang::NamedDecl *decl,
                              clang::NamedDecl *defDecl) {
  assert(!keyName.empty() && "KeyName should not be empty!");
  db.try_emplace(keyName, std::make_unique<ObjInfo>(keyName, decl, defDecl));
  if (decl->getIdentifier()) {
    auto plainName = decl->getQualifiedNameAsString();
    manglings[plainName].insert(keyName);
  }
}

void CodeDB::registerDecl(clang::ASTUnit const &unit, clang::NamedDecl *decl,
                          clang::NamedDecl *defDecl) {
  registerDeclImpl(unit, getKeyName(decl), decl, defDecl);
}

void CodeDB::registerDecl(clang::ASTUnit const &unit, clang::RecordDecl *decl,
                          clang::RecordDecl *defDecl) {
  registerDeclImpl(unit, getKeyName(decl), decl, defDecl);
  // For templated decls, we need to visit all specialized methods here.
  if (auto cxxRD = llvm::dyn_cast<clang::CXXRecordDecl>(decl)) {
    auto classtemp = cxxRD->getDescribedClassTemplate();
    if (!classtemp)
      return;
    for (auto spec : classtemp->specializations()) {
      for (auto method : spec->methods())
        registerDecl(unit, method, method->getDefinition());
    }
  }
}

void CodeDB::registerDecl(clang::ASTUnit const &unit, clang::FunctionDecl *decl,
                          clang::FunctionDecl *defDecl) {
  for (auto redecl : decl->redecls()) {
    auto key = getKeyName(redecl);

    if (auto tmpDecl = redecl->getDescribedFunctionTemplate())
      registerDecl(unit, tmpDecl, tmpDecl);
    else if (key.empty())
      // Sometimes we end up picking hidden function decls.
      // We should just try to skip them for now...
      continue;
    else
      registerDeclImpl(unit, key, redecl, redecl->getDefinition());
  }
}

void CodeDB::registerDecl(clang::ASTUnit const &unit,
                          clang::FunctionTemplateDecl *decl,
                          clang::FunctionTemplateDecl *defDecl) {
  for (auto specs : decl->specializations())
    registerDeclImpl(unit, getKeyName(specs), specs, specs->getDefinition());
}
