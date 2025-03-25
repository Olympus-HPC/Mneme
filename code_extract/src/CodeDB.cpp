#include "CodeDB.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Mangle.h"

#include <cassert>

std::string CodeDB::getKeyName(clang::NamedDecl const *decl) {
  if (llvm::dyn_cast<clang::TypeDecl>(decl)) {
    std::string prefix; 
    if (decl->getKind() == clang::Decl::Kind::Typedef)
      prefix = "typedef ";
    return prefix + decl->getQualifiedNameAsString();
  }
  else
    return clang::ASTNameGenerator(decl->getASTContext()).getName(decl);
}

void CodeDB::registerDeclImpl(clang::ASTUnit const &unit, std::string keyName,
                              clang::NamedDecl *decl,
                              clang::NamedDecl *defDecl) {
  assert(!keyName.empty() && "KeyName should not be empty!");
  db.try_emplace(keyName,
                 std::make_unique<ObjInfo>(keyName, decl, defDecl));
  auto plainName = decl->getQualifiedNameAsString();
  manglings[plainName].insert(keyName);
}

void CodeDB::registerDecl(clang::ASTUnit const &unit, clang::NamedDecl *decl,
                          clang::NamedDecl *defDecl) {
  registerDeclImpl(unit, getKeyName(decl), decl, defDecl);
}

void CodeDB::registerDecl(clang::ASTUnit const &unit, clang::FunctionDecl *decl,
                          clang::FunctionDecl *defDecl) {
  if (clang::FunctionTemplateDecl *tmpDecl =
          decl->getDescribedFunctionTemplate()) {
    registerDecl(unit, tmpDecl, tmpDecl);
  } else {
    for (auto redecl : decl->redecls()) {
      auto key = getKeyName(redecl);
      // Sometimes we end up picking hidden function decls.
      // We should just try to skip them for now...
      if (key.empty())
        continue;
      registerDeclImpl(unit, key, redecl, redecl->getDefinition());
    }
  }
}

void CodeDB::registerDecl(clang::ASTUnit const &unit,
                          clang::FunctionTemplateDecl *decl,
                          clang::FunctionTemplateDecl *defDecl) {
  for (auto specs : decl->specializations())
    registerDeclImpl(unit, getKeyName(specs), specs, specs->getDefinition());
}
