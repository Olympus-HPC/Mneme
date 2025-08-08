#include "Visitor.h"
#include "CodeDB.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Frontend/ASTUnit.h"

#include <type_traits>

namespace helper {
std::string locToIncFile(clang::SourceLocation sloc,
                         clang::ASTContext const &ctx) {
  auto &mgr = ctx.getSourceManager();
  auto Entry = mgr.getFileEntryForID(mgr.getFileID(sloc));
  if (!Entry)
    return "";
  return Entry->tryGetRealPathName().str();
}

bool isIncludeExternal(std::string const &incFile, CodeDB const &codedb) {
  return incFile.find(codedb.projPath) == std::string::npos;
}

template <typename T>
void storeDecl(T *decl, clang::ASTUnit const &unit, CodeDB &cdb) {
  constexpr bool isFunctionDecl = std::is_same_v<T, clang::FunctionDecl>;
  std::string srcDeclFile =
      locToIncFile(decl->getLocation(), unit.getASTContext());
  if constexpr (isFunctionDecl) {
    // If function template, look at template declaration loc and not
    // instantiation loc.
    if (auto tmpDecl = decl->getPrimaryTemplate()) {
      srcDeclFile = locToIncFile(tmpDecl->getLocation(), unit.getASTContext());
    }
  }
  // If location is external but not a function decl, dont store it
  // We need to store external function decls as they may be requested for
  // extraction.
  if (isIncludeExternal(srcDeclFile, cdb) &&
      !(cdb.includeExternals && isFunctionDecl))
    return;

  std::string keyName = CodeDB::getKeyName(decl);
  T *defDecl = decl->getDefinition();

  if (!cdb.isRegistered(keyName)) {
    cdb.registerDecl(unit, decl, defDecl);
  } else {
    if (defDecl)
      cdb.addDefinitionDecl(keyName, defDecl);
  }
}

// This might not be efficient but we still keep it.
void visitNamespaceChain(clang::Decl const *decl, VisitManager &vm) {
  auto *ctx = decl->getDeclContext();
  while (ctx) {
    if (auto const *ns = llvm::dyn_cast<clang::NamespaceDecl>(ctx)) {
      if (ns->isAnonymousNamespace())
        continue;
      vm.addIfNamespaceAliased(ns->getNameAsString());
    }
    ctx = ctx->getParent();
  }
}

std::tuple<clang::Decl const *, bool>
visit(clang::NamedDecl const *decl, VisitManager &vm, CodeDB const &cdb) {
  std::string keyName = CodeDB::getKeyName(decl);

  if (vm.isVisited(keyName))
    return {vm.getVisitedObj(keyName)->getDefinition(), false};

  helper::visitNamespaceChain(decl, vm);

  auto incFile = locToIncFile(decl->getLocation(), decl->getASTContext());
  if (isIncludeExternal(incFile, cdb))
    return {decl, false};

  if (!cdb.isRegistered(keyName))
    decl->dump();
  assert(cdb.isRegistered(keyName) &&
         "All decl to visit should be registered!");
  auto objInfo = cdb.getObjInfoOrNull(keyName);

  vm.markVisited(keyName, objInfo);
  // register source for includes
  vm.registerInclude(incFile);
  auto defDecl = objInfo->getDefinition();

  // Lookup definition from database
  return {defDecl, true};
}

/// @brief Checks if the given input is either of the types specifed in the
/// variadic template parameters.
/// @tparam A Types to check against.
/// @tparam ...B Types to check against.
/// @param expr Expression to check for.
/// @return True if the input expression is of any one type specifed in the
/// template parameter. False otherwise.
template <class A, class... B> bool isOneOf(clang::Expr const *expr) {
  if constexpr (sizeof...(B) == 0)
    return false;
  else {
    if (llvm::dyn_cast<A>(expr))
      return true;
    else
      return isOneOf<B...>(expr);
  }
}

bool isGlobalVar(clang::VarDecl const *decl) {
  return !decl->isStaticLocal() && decl->hasGlobalStorage();
}

clang::QualType getUnderlyingType(clang::QualType const &qualType) {
  auto type = qualType.getLocalUnqualifiedType();
  clang::QualType qt;
  if (auto pointerType = type->getAs<clang::PointerType>())
    qt = pointerType->getPointeeType();
  else if (auto arrayType = llvm::dyn_cast<clang::ArrayType>(type))
    qt = arrayType->getElementType();
  else if (auto referenceType = type->getAs<clang::ReferenceType>())
    qt = referenceType->getPointeeType();
  else
    return type;

  return getUnderlyingType(qt);
}

// Use the fact that builtin functions are typically prepended with "__"
bool isPotentialBuiltinByName(std::string const &name) {
  return name.size() > 2 && '_' == name[0] && '_' == name[1];
}

void typeVisitorHelper(clang::QualType qt, VisitManager &vm,
                       CodeDB const &codedb, bool isWithinTypedef = false);
void handleRecordDecl(clang::RecordDecl const *recordDecl, VisitManager &vm,
                      CodeDB const &codedb, bool dontEmitDecl = false) {
  // If externally defined (or built-in), do not include def as we will include
  // the file itself.
  if (isPotentialBuiltinByName(recordDecl->getNameAsString()) ||
      recordDecl->isImplicit())
    return;

  // First visit all field types
  for (auto field : recordDecl->fields())
    typeVisitorHelper(field->getType(), vm, codedb);

  // We will typically not find RecordDecls within function bodies or init
  // expressions. Hence, we need to visit them when we encounter either their
  // var decl or static function call (unsupported as of yet).
  auto [defDecl, doVisit] = helper::visit(recordDecl, vm, codedb);
  if (doVisit && !dontEmitDecl) {
    vm.registerDecl(static_cast<clang::RecordDecl const *>(defDecl));
  }

  // Also we do not want to visit anything else from here as we will visit the
  // function calls separately
}

void typeVisitorHelper(clang::QualType qt, VisitManager &vm,
                       CodeDB const &codedb, bool isWithinTypedef) {
  if (!qt.getTypePtrOrNull())
    return;

  auto cannonType = getUnderlyingType(qt);
  if (auto typedefType = cannonType->getAs<clang::TypedefType>()) {
    // recursively visit underlying type(defs).
    // We should hopefully create a handle typedef function that does not
    // revisit typedef chains again if visited before.
    auto typDecl = typedefType->getDecl();

    if (isPotentialBuiltinByName(typDecl->getNameAsString()))
      return;

    typeVisitorHelper(typDecl->getUnderlyingType(), vm, codedb, true);
    auto [defDecl, doVisit] = visit(typDecl, vm, codedb);

    if (doVisit && defDecl->isDefinedOutsideFunctionOrMethod())
      vm.registerDecl(static_cast<clang::TypedefNameDecl const *>(defDecl));
  } else if (auto decl = cannonType->getAsCXXRecordDecl()) {
    if (isWithinTypedef || !decl->isDefinedOutsideFunctionOrMethod())
      vm.markIndirectDef(decl);
    handleRecordDecl(decl, vm, codedb,
                     isWithinTypedef && decl->isThisDeclarationADefinition());
  }
}
} // namespace helper

bool CodeExtractVisitor::VisitVarDecl(clang::VarDecl *decl) {
  if (!helper::isGlobalVar(decl))
    return true;
  helper::storeDecl(decl, unit, codedb);
  return true;
}

bool CodeExtractVisitor::VisitFunctionDecl(clang::FunctionDecl *decl) {
  helper::storeDecl(decl, unit, codedb);
  return true;
}

bool CodeExtractVisitor::VisitRecordDecl(clang::RecordDecl *decl) {
  helper::storeDecl(decl, unit, codedb);
  return true;
}

bool CodeExtractVisitor::VisitTypedefNameDecl(clang::TypedefNameDecl *decl) {
  if (codedb.isRegistered(CodeDB::getKeyName(decl)))
    return true;
  codedb.registerDecl(unit, decl, decl);
  return true;
}

bool CodeExtractVisitor::VisitNamespaceAliasDecl(
    clang::NamespaceAliasDecl *decl) {
  codedb.addNamespaceAlias(decl);
  return true;
}

bool CodeExtractVisitor::VisitEnumDecl(clang::EnumDecl *decl) {
  helper::storeDecl(decl, unit, codedb);
  return true;
}

bool MatchVisitor::VisitCXXConstructExpr(clang::CXXConstructExpr *expr) {
  auto ctor = expr->getConstructor();
  if (helper::isPotentialBuiltinByName(ctor->getNameAsString()))
    return true;
  helper::handleRecordDecl(ctor->getParent(), vm, codedb);
  return true;
}

bool MatchVisitor::VisitTypeLoc(clang::TypeLoc typ) {
  helper::typeVisitorHelper(typ.getType(), vm, codedb);
  return true;
}

bool MatchVisitor::VisitDeclRefExpr(clang::DeclRefExpr *declRef) {
  // Need to filter out global varDeclRef...
  auto varDecl = llvm::dyn_cast<clang::VarDecl>(declRef->getDecl());
  if (!varDecl)
    return true;

  // Need to filter out cuda internals, for now we check if either the variable
  // name or variable type name is potentially builtin. OR if the variable decl
  // itself has a builtin attribute.
  bool isPotentialBuiltin =
      helper::isPotentialBuiltinByName(varDecl->getNameAsString()) ||
      helper::isPotentialBuiltinByName(
          varDecl->getType().getUnqualifiedType().getAsString()) ||
      varDecl->hasAttr<clang::BuiltinAttr>();
  if (!helper::isGlobalVar(varDecl) || isPotentialBuiltin)
    return true;

  auto [decl, visitBody] = helper::visit(varDecl, vm, codedb);
  auto defDecl = static_cast<clang::VarDecl const *>(decl);
  if (!visitBody)
    return true;

  TraverseVarDecl(const_cast<clang::VarDecl *>(defDecl));

  // If this vardecl is locally but extern'd we don't visit its init or emit it.
  if (defDecl->isLocalExternDecl())
    return true;

  assert((defDecl->hasDefinition() || defDecl->hasExternalStorage()) &&
         "We should have seen this non-extern variable's decl before!");

  /// FIXIT: Replace with better design...
  auto varInit = const_cast<clang::VarDecl *>(defDecl)->getInit();
  // If the init expression is not a literal, we should visit it to resolve
  // dependencies. Obviously this list of literals is not exhaustive.
  if (varInit &&
      !helper::isOneOf<clang::CXXBoolLiteralExpr, clang::CharacterLiteral,
                       clang::FixedPointLiteral, clang::FloatingLiteral,
                       clang::IntegerLiteral, clang::StringLiteral>(varInit))
    TraverseStmt(varInit);

  vm.registerDecl(defDecl);
  return true;
}

bool MatchVisitor::VisitCallExpr(clang::CallExpr *callExpr) {
  auto decl = callExpr->getDirectCallee();
  auto incFile =
      helper::locToIncFile(decl->getLocation(), decl->getASTContext());
  if (!decl || (helper::isPotentialBuiltinByName(decl->getNameAsString()) &&
                helper::isIncludeExternal(incFile, codedb)))
    return true;

  clang::CXXRecordDecl *parentDecl = nullptr;
  /// FIXME: Make more generic to handle other record types.
  if (decl->isCXXClassMember()) {
    parentDecl = static_cast<clang::CXXMethodDecl *>(decl)->getParent();
    // If parent decl is implicit, do not visit as callexpr maybe lambda
    // Or if function is default, do not visit
    if (parentDecl->isImplicit() || decl->isDefaulted())
      return true;
  }

  // Redundant but better to check here than later
  if (vm.isVisited(CodeDB::getKeyName(decl))) {
    vm.fwdDeclFuncDecl(decl);
    return true;
  }

  auto [DBdecl, visitBody] = helper::visit(decl, vm, codedb);
  auto defDecl = static_cast<clang::FunctionDecl const *>(DBdecl);

  // If function is a call to a template instantiation, we need to save the
  // includes for it at the point of instantiation (as well).
  if (decl->isFunctionTemplateSpecialization())
    vm.registerInclude(incFile);

  if (!visitBody || !defDecl->hasBody())
    return true;

  // First visit dependent calls
  // Only internal state is modified, so we can cast away the const-ness here.
  TraverseFunctionDecl(const_cast<clang::FunctionDecl *>(defDecl));

  // Then register
  vm.registerDecl(defDecl);

  return true;
}
