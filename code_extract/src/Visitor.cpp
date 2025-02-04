#include "Visitor.h"
#include "CodeDB.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Frontend/ASTUnit.h"

#include <type_traits>

namespace helper {
std::string locToIncFile(clang::SourceLocation sloc,
                         clang::ASTContext const &ctx) {
  auto declFile = sloc.printToString(ctx.getSourceManager());
  return declFile.substr(0, declFile.find_first_of(':'));
}

bool isIncludeExternal(std::string const &incFile, CodeDB const &codedb) {
  /// FIXME: For now we use a trick to figure out if the include is system-wide
  /// or local. Typically local includes will show up as relative paths in the
  /// code's source location. Eventually we should make this check more robust.
  return incFile[0] == '/' &&
         incFile.find(codedb.projPath) == std::string::npos;
}

bool checkPotentialInclude(clang::NamedDecl const *decl, VisitManager &vm,
                           CodeDB const &codedb) {
  auto incFile = locToIncFile(decl->getLocation(), decl->getASTContext());
  if (isIncludeExternal(incFile, codedb))
    return vm.registerInclude(incFile);
  else
    return false;
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
  bool isExternal = isIncludeExternal(srcDeclFile, cdb);
  // If location is external but not a function decl, dont store it
  // We need to store external function decls as they may be requested for
  // extraction.
  if (isExternal && !(cdb.includeExternals && isFunctionDecl))
    return;

  std::string keyName = CodeDB::getKeyName(decl);
  T *defDecl = decl->getDefinition();

  if (!cdb.isRegistered(keyName)) {
    cdb.registerDecl(unit, decl, defDecl);
  } else {
    if (defDecl)
      cdb.addDefinitionDecl(keyName, defDecl);
  }

  if (isExternal)
    cdb.addExtSourceToSourceName(decl->getQualifiedNameAsString(), srcDeclFile);
}

template <typename T>
std::tuple<T const *, bool> visitAndRegister(clang::NamedDecl const *decl,
                                             VisitManager &vm,
                                             CodeDB const &cdb) {
  std::string keyName = CodeDB::getKeyName(decl);
  if (vm.isVisited(keyName))
    return {static_cast<T const *>(vm.getVisitedObj(keyName)->getDefiniton()),
            false};

  if (!cdb.isRegistered(keyName))
    decl->dump();
  assert(cdb.isRegistered(keyName) &&
         "All decl to visit should be registered!");
  auto objInfo = cdb.getObjInfoOrNull(keyName);

  // Lookup definition from database
  auto defDecl = static_cast<T const *>(objInfo->getDefiniton());
  vm.markVisited(keyName, objInfo);
  vm.registerDecl(defDecl);

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

clang::QualType getUnderlyingType(clang::QualType const &type) {
  clang::QualType qt;
  if (auto pointerType = type->getAs<clang::PointerType>())
    qt = pointerType->getPointeeType();
  else if (type->isArrayType())
    qt = llvm::cast<clang::ArrayType>(type)->getElementType();
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

void handleRecordDecl(clang::RecordDecl const *recordDecl, VisitManager &vm,
                      CodeDB const &codedb) {
  // If externally defined (or built-in), do not include def as we will include
  // the file itself.
  if (checkPotentialInclude(recordDecl, vm, codedb) ||
      isPotentialBuiltinByName(recordDecl->getNameAsString()) ||
      recordDecl->isImplicit())
    return;

  // We will typically not find RecordDecls within function bodies or init
  // expressions. Hence, we need to visit them when we encounter either their
  // var decl or static function call (unsupported as of yet).
  helper::visitAndRegister<clang::RecordDecl>(recordDecl, vm, codedb);

  // Also we do not want to visit anything else from here as we will visit the
  // function calls separately
}

// A little bit of code duplication for clarity.
void handleTypedefs(clang::TypedefType const *typ, VisitManager &vm,
                    CodeDB const &codedb) {
  if (!typ)
    return;

  auto typDecl = typ->getDecl();
  // If externally defined (or built-in), do not include def as we will
  // include the file itself.
  if (checkPotentialInclude(typDecl, vm, codedb) ||
      isPotentialBuiltinByName(typDecl->getNameAsString()))
    return;

  handleTypedefs(typDecl->getUnderlyingType()->getAs<clang::TypedefType>(), vm,
                 codedb);

  visitAndRegister<clang::TypedefNameDecl>(typDecl, vm, codedb);
}

void handleVarDecl(clang::QualType qt, VisitManager &vm, CodeDB const &codedb) {
  auto cannonType = getUnderlyingType(qt);
  clang::RecordDecl const *decl = cannonType->getAsRecordDecl();
  if (!decl) {
    auto typedefType = cannonType->getAs<clang::TypedefType>();
    // recursively visit underlying type(defs).
    // We should hopefully
    // create a handle typedef function that does not revisit typedef chains
    // again if visited before.
    handleTypedefs(typedefType, vm, codedb);
  } else
    handleRecordDecl(decl, vm, codedb);
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

/// FIXME: We do not need to cache these as typically defs are together with
/// decls.
bool CodeExtractVisitor::VisitTypedefNameDecl(clang::TypedefNameDecl *decl) {
  std::string keyName = CodeDB::getKeyName(decl);
  if (codedb.isRegistered(keyName))
    return true;
  codedb.registerDecl(unit, decl, decl);
  return true;
}

bool MatchVisitor::VisitCXXConstructExpr(clang::CXXConstructExpr *expr) {
  auto ctor = expr->getConstructor();
  if (helper::isPotentialBuiltinByName(ctor->getNameAsString()))
    return true;
  helper::handleRecordDecl(ctor->getParent(), vm, codedb);
  return true;
}

bool MatchVisitor::VisitDeclRefExpr(clang::DeclRefExpr *declRef) {
  // Need to filter out global varDeclRef...
  auto varDecl = llvm::dyn_cast<clang::VarDecl>(declRef->getDecl());
  if (!varDecl)
    return true;

  // Even if the vardecls are not interesting, visit their type.
  helper::handleVarDecl(declRef->getType(), vm, codedb);

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

  auto [defDecl, visitBody] =
      helper::visitAndRegister<clang::VarDecl>(varDecl, vm, codedb);
  if (!visitBody)
    return true;

  assert(
      defDecl->hasDefinition() &&
      "We should have seen this variable's decl before unless it is external!");

  /// FIXIT: Replace with better design...
  auto varInit = const_cast<clang::VarDecl *>(defDecl)->getInit();
  // If the init expression is not a literal, we should visit it to resolve
  // dependencies. Obviously this list of literals is not exhaustive.
  if (varInit &&
      !helper::isOneOf<clang::CXXBoolLiteralExpr, clang::CharacterLiteral,
                       clang::FixedPointLiteral, clang::FloatingLiteral,
                       clang::IntegerLiteral, clang::StringLiteral>(varInit))
    vm.addToVisit(varInit);

  return true;
}

bool MatchVisitor::VisitCallExpr(clang::CallExpr *callExpr) {
  auto decl = callExpr->getDirectCallee();
  if (!decl || helper::checkPotentialInclude(decl, vm, codedb) ||
      helper::isPotentialBuiltinByName(decl->getNameAsString()))
    return true;

  clang::CXXRecordDecl *parentDecl = nullptr;
  /// FIXME: Make more generic to handle other record types.
  if (decl->isCXXClassMember()) {
    parentDecl = static_cast<clang::CXXMethodDecl *>(decl)->getParent();
    // If parent decl is implicit, do not visit as callexpr maybe lambda
    if (parentDecl->isImplicit())
      return true;
  }

  auto [defDecl, visitBody] =
      helper::visitAndRegister<clang::FunctionDecl>(decl, vm, codedb);
  if (!visitBody || !defDecl->hasBody())
    return true;
  vm.addToVisit(defDecl->getBody());

  // Handle function param var decls
  // Also, dont visit the parent decl here, we either have visited it or will
  // eventually...
  VisitParams(defDecl, false);

  if (parentDecl && decl->isStatic())
    helper::handleRecordDecl(parentDecl, vm, codedb);

  return true;
}

void MatchVisitor::VisitParams(clang::FunctionDecl const *defDecl,
                               bool isRecordMember) {
  // If function is a record member, visit its record.
  if (isRecordMember)
    helper::handleRecordDecl(
        static_cast<clang::CXXMethodDecl const *>(defDecl)->getParent(), vm,
        codedb);

  for (auto param_it : defDecl->parameters())
    helper::handleVarDecl(param_it->getType(), vm, codedb);
}

void MatchVisitor::VisitTemplateParams(clang::FunctionDecl const *defDecl) {
  auto tmpSpec = defDecl->getTemplateSpecializationInfo();
  if (tmpSpec) {
    auto tmpArgs = tmpSpec->TemplateArguments->asArray();
    for (auto arg : tmpArgs)
      helper::handleVarDecl(arg.getAsType(), vm, codedb);
  }
}
