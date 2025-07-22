#include "VisitManager.h"
#include "Visitor.h"

#include <string>
#include <unordered_map>
#include <vector>

#include "clang/AST/Attrs.inc"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/LLVM.h"
#include "llvm/Support/raw_ostream.h"

namespace helper {

class LambdaCallback : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  clang::LambdaExpr const *lambdaExpr = nullptr;

  void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) final {
    if (auto lmbdExpr =
            Result.Nodes.getNodeAs<clang::LambdaExpr>("lambdaExpr")) {
      assert(lambdaExpr == nullptr &&
             "Should only match one lambda expression.");
      lambdaExpr = lmbdExpr;
    }
  }
};

clang::QualType stripRefs(clang::QualType type) {
  if (auto refType = type->getAs<clang::ReferenceType>())
    return refType->getPointeeType();
  else
    return type;
}

std::string getParamDeclAsString(std::string const &typeString,
                                 std::string const &name,
                                 std::string const &init = "") {
  auto inString = typeString;
  // Clang ends up printing the C-style bool so we need to replace that.
  size_t boolPos = inString.find("_Bool");
  if (boolPos != std::string::npos)
    inString = inString.replace(boolPos, 5, "bool");

  auto prefixEnd = std::min(inString.find_last_of(')'), inString.size());
  std::string prefix = inString.substr(0, prefixEnd);
  std::string suffix = inString.substr(prefixEnd);
  std::string decl = prefix + " " + name + suffix;
  if (!init.empty())
    decl += " = " + init;
  return decl + ";\n";
}

void buildExplicitTempSpec(
    clang::ArrayRef<clang::TemplateArgument> const &tmpArgs,
    llvm::raw_string_ostream &ss) {
  ss << "< ";
  for (auto &arg : tmpArgs) {
    bool stop = false;
    switch (arg.getKind()) {
    case clang::TemplateArgument::ArgKind::Integral: {
      ss << arg.getAsIntegral() << ",";
      break;
    }
    case clang::TemplateArgument::ArgKind::Type: {
      auto argType = arg.getAsType();
      stop = argType->hasUnnamedOrLocalType();
      if (stop)
        break;
      ss << argType.getAsString() << ",";
      break;
    }
    }
    if (stop)
      break;
  }
  ss.str().back() = '>';
}

void buildExplicitTempSpec(clang::FunctionDecl const *fn,
                           llvm::raw_string_ostream &ss) {
  if (auto tmpSpec = fn->getTemplateSpecializationInfo())
    buildExplicitTempSpec(tmpSpec->TemplateArguments->asArray(), ss);
}

void buildExplicitTempSpec(clang::CXXRecordDecl const *decl,
                           llvm::raw_string_ostream &ss) {
  if (auto recordTmp =
          llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(decl))
    buildExplicitTempSpec(recordTmp->getTemplateArgs().asArray(), ss);
}

void buildCallExpr(clang::ArrayRef<clang::ParmVarDecl *> const &parameters,
                   llvm::raw_string_ostream &ss,
                   std::string const &paramPrefix = "p") {
  int paramCount = 0;
  ss << "( ";
  for (auto param : parameters) {
    std::string paramName = paramPrefix + std::to_string(paramCount);
    auto type = param->getType();
    if (type->isRValueReferenceType()) {
      auto strippedType =
          helper::stripRefs(type).getUnqualifiedType().getAsString();
      if (type->hasUnnamedOrLocalType())
        ss << "std::forward<decltype(" << paramName << ")>(" << paramName
           << ")";
      else
        ss << "std::forward<" << strippedType << ">(" << paramName << ")";
    } else
      ss << paramName;
    ss << ",";
    paramCount++;
  }
  ss.str().back() = ')';
}

// Clang's print forcibly does not print attributes for function
// templates, we need to override this behaviour and manually print them
// ourselves.
void print(llvm::raw_ostream &ss, clang::Decl const *decl, bool terse = false) {
  clang::PrintingPolicy pp(decl->getLangOpts());
  pp.TerseOutput = terse;
  std::string outString;
  llvm::raw_string_ostream out(outString);

  decl->print(out, pp);
  auto fnDecl = decl->getAsFunction();
  if (decl->getKind() == clang::Decl::Kind::Var)
    out << ";";
  else if (fnDecl && fnDecl->getDescribedFunctionTemplate() ||
           fnDecl->isFunctionTemplateSpecialization()) {
    std::string attrs;
    llvm::raw_string_ostream attrsStream(attrs);
    auto &Attrs = fnDecl->getAttrs();
    for (auto *A : Attrs) {
      if (A->isInherited() || A->isImplicit())
        continue;
      A->printPretty(attrsStream, pp);
      attrsStream << ' ';
    }
    auto fnName = fnDecl->getName();
    outString.insert(outString.find(fnName), attrs);
  }

  ss << outString;
}

void makeCUDACompatible(std::string &code) {
  // Cuda C/C++ does not allow pragma parameters in parenthesis so we need to
  // remove and redo those lines
  size_t pos = 0;
  while (pos != std::string::npos) {
    auto from = code.find("#pragma", pos);
    if (from == std::string::npos)
      break;
    auto to = code.find("\n", from);
    auto pragma = code.substr(from, to - from);

    std::replace(pragma.begin(), pragma.end(), ',', ' ');
    std::replace(pragma.begin(), pragma.end(), '(', ' ');
    std::replace(pragma.begin(), pragma.end(), ')', ' ');

    if (auto enPos = pragma.find("enable"))
      pragma.replace(enPos, 6, " ");

    code = code.replace(from, to - from, pragma);
    pos = to + 1;
  }
}

/// FIXME: Move into generic utils namespace
extern clang::QualType getUnderlyingType(clang::QualType const &type);
extern std::string locToIncFile(clang::SourceLocation sloc,
                                clang::ASTContext const &ctx);
extern bool isIncludeExternal(std::string const &incFile, CodeDB const &codedb);
} // namespace helper

void VisitManager::markIndirectDef(clang::TagDecl const *decl) {
  hasIndirectDef.insert(decl);
}

void VisitManager::registerDecl(clang::NamedDecl const *decl) {
  declRefs.push_back(decl);
}

void VisitManager::registerDecl(clang::CXXRecordDecl const *decl) {
  // We should not emit record decls that are part of typedefs as they will be
  // emitted alongside the typedef.
  if (decl->getTypedefNameForAnonDecl())
    return;

  if (auto classtmp = decl->getDescribedClassTemplate())
    tagDecls.push_back(classtmp);
  else
    tagDecls.push_back(decl);
}

void VisitManager::registerDecl(clang::TypedefNameDecl const *decl) {
  auto underlyingTag = decl->getUnderlyingType()->getAsTagDecl();
  if (underlyingTag) {
    auto file = helper::locToIncFile(underlyingTag->getLocation(),
                                     underlyingTag->getASTContext());
    // Ugly fix, we don't want to expand typedefs over externals.
    if (helper::isIncludeExternal(file, db)) {
      tagDecls.push_back(decl);
      return;
    }
  }

  /// FIXME: Separate typedefs over struct defs, handle them differently.
  // We always expand this with the full definition of the aliased type.
  typedefDecls.push_back(decl);
}

void VisitManager::registerDecl(clang::FunctionDecl const *decl) {
  // Do not emit inlined functions
  bool isClassMember = decl->isCXXClassMember();
  if (isClassMember && decl->isInlined())
    return;

  if (auto tmpDecl = decl->getPrimaryTemplate()) {
    // If it is an explicit template spec, also add the definition of that
    // specialization.
    if (decl->getTemplateSpecializationInfo()->isExplicitSpecialization()) {
      fwdDecls.insert(tmpDecl);
      declRefs.push_back(decl);
    } else if (tempDecls.find(tmpDecl) == tempDecls.end()) {
      tempDecls.insert(tmpDecl);
      declRefs.push_back(tmpDecl);
    }
  } else
    declRefs.push_back(decl);
}

void VisitManager::fwdDeclFuncDecl(clang::FunctionDecl const *decl) {
  if (decl->isCXXClassMember())
    return; // Even class members can have circular dependencies but ignore
            // here.
  if (auto tmpDecl = decl->getPrimaryTemplate())
    fwdDecls.insert(tmpDecl);
  else
    fwdDecls.insert(decl);
}

void VisitManager::registerInclude(std::string const &includePath) {
  auto [id, needsExplicitInclude] = db.includes.getIDFromFile(includePath);
  if (needsExplicitInclude)
    db.includes.getIncludes(id, includes);
}

bool VisitManager::isVisited(std::string const &name) {
  return visitedNodes.find(name) != visitedNodes.end();
}

ObjInfo const *VisitManager::getVisitedObj(std::string const &name) {
  return visitedNodes.at(name);
}

void VisitManager::markVisited(std::string const &name,
                               ObjInfo const *objInfo) {
  visitedNodes.try_emplace(name, objInfo);
}

template <typename T>
void VisitManager::fillParams(std::string const &prefix, T *begin, T *end,
                              MatchVisitor &mv) {
  int idx = 0;
  for (auto paramIt = begin; paramIt != end; paramIt++) {
    clang::ValueDecl const *expr = nullptr;
    std::string key = "p" + prefix + std::to_string(idx++);
    if constexpr (std::is_same_v<T, clang::LambdaCapture const>) {
      expr = paramIt->getCapturedVar();
      // We need to restore the captures with the same name, hence we do not
      // follow the p<position> naming convention.
      key = expr->getNameAsString();
    } else {
      expr = *paramIt;
    }
    auto type = helper::getUnderlyingType(expr->getType());
    auto recordDecl = type->getAsCXXRecordDecl();
    // If expression is lambda, save that...
    if (recordDecl && type->hasUnnamedOrLocalType()) {
      // Could possibly be lambda object,
      // find underlying expression.
      using namespace clang::ast_matchers;
      StatementMatcher lmbdExpr =
          lambdaExpr(hasType(asString(type.getAsString()))).bind("lambdaExpr");
      MatchFinder finder;
      helper::LambdaCallback callback;
      finder.addMatcher(lmbdExpr, &callback);
      finder.matchAST(recordDecl->getASTContext());

      auto lmbExpr = callback.lambdaExpr;
      assert(lmbExpr && "Could not find lambda expression!");

      mv.TraverseStmt(lmbExpr->getBody());
      std::string lmbBody, lmbAttrs;
      llvm::raw_string_ostream stream(lmbBody);
      llvm::raw_string_ostream attrStream(lmbAttrs);

      for (auto attr : lmbExpr->getCallOperator()->getAttrs())
        attr->printPretty(attrStream, recordDecl->getLangOpts());
      lmbExpr->printPretty(stream, nullptr, recordDecl->getLangOpts());
      if (!lmbAttrs.empty())
        lmbBody.insert(lmbBody.find(']') + 1, lmbAttrs);
      params_expr.emplace_back(key, lmbBody);

      fillParams(key, lmbExpr->capture_begin(), lmbExpr->capture_end(), mv);
    } else if (recordDecl) {
      clang::CXXConstructorDecl *ctor = nullptr;
      bool hasNonDeletedDefault = false;
      for (auto constructor : recordDecl->ctors()) {
        if (constructor->isDeleted() || constructor->isCopyOrMoveConstructor())
          continue;
        ctor = constructor;
        if (constructor->isDefaultConstructor()) {
          hasNonDeletedDefault = true;
          break;
        }
      }

      if (!ctor || !ctor->getNumParams() || hasNonDeletedDefault) {
        params_decl.emplace_back(key, expr);
        continue;
      }

      std::string ctorCall;
      llvm::raw_string_ostream stream(ctorCall);
      stream << recordDecl->getQualifiedNameAsString();
      helper::buildExplicitTempSpec(recordDecl, stream);
      helper::buildExplicitTempSpec(ctor, stream);
      helper::buildCallExpr(ctor->parameters(), stream, key + "_");
      params_expr.emplace_back(key, ctorCall);

      fillParams(key.substr(1) + "_", ctor->param_begin(), ctor->param_end(),
                 mv);
    } else {
      params_decl.emplace_back(key, expr);
    }
    /// FIXME: We can do more here, like also handling function pointers like we
    /// do lambdas...
  }
}

void VisitManager::registerParameterPrologue(MatchVisitor &mv, ObjInfo *fnObj) {
  clang::FunctionDecl *fnDecl = fnObj->getDefinition()->getAsFunction();

  if (!fnDecl->getNumParams())
    return;

  fillParams("", fnDecl->param_begin(), fnDecl->param_end(), mv);
}

void VisitManager::emitStandaloneFile(std::string &output, bool emitRR,
                                      std::string const &configString) {
  auto body =
      static_cast<clang::FunctionDecl const *>(primaryFn.getDefinition());
  bool cudaKernel = body->hasAttr<clang::CUDAGlobalAttr>();
  bool emitRRHooks = cudaKernel && emitRR;

  llvm::raw_string_ostream ss(output);

  for (auto &incID : includes) {
    auto inc = db.includes.getFileFromID(incID);
    // For now we do not support hip
    if (inc.find("hip_runtime") != std::string::npos)
      continue;
    ss << "#include ";
    if (inc.find('.') == std::string::npos)
      ss << "<" << inc << ">";
    else
      ss << "\"" << inc << "\"";
    ss << "\n";
  }
  if (emitRRHooks)
    ss << "#include \"RRHooks.h\"\n";
  ss << "\n";

  for (auto &tags : tagDecls) {
    if (hasIndirectDef.find(tags) != hasIndirectDef.end())
      continue;
    tags->print(ss);
    ss << ";\n\n";
  }

  clang::PrintingPolicy tagPolicy(body->getLangOpts());
  tagPolicy.IncludeTagDefinition = true;
  for (auto &tags : typedefDecls) {
    tags->print(ss, tagPolicy);
    ss << ";\n\n";
  }

  // Emit all forward decls
  for (auto &decl : fwdDecls) {
    helper::print(ss, decl, true);
    // Since these are just fwd decls, add a semicolon
    ss << ";\n\n";
  }
  ss << '\n';

  // Emit all other declrefs
  for (auto decl : declRefs) {
    helper::print(ss, decl);
    ss << "\n\n";
  }

  // Building main
  ss << "int main(int argc, char *argv[]) {\n";

  if (emitRRHooks) {
    // First add the prologue
    ss << "init_RR(";
    ss << "\"" << primaryFn.keyName // Function key names will always be
                                    // their mangled names...
       << "\"";
    ss << ", argc, argv);\n";
  }

  if (cudaKernel)
    // If we have a cuda kernel, we should add the config string...
    ss << "dim3 grid;\n"
       << "dim3 block;\n";

  if (emitRRHooks) {
    // ...and load it from RR
    ss << "init_dims(\"Grid\", grid);\n"
       << "init_dims(\"Block\", block);\n";
  }

  // Build parameter decl prologue
  // First emit all params without any initializers.
  bool hasFwdTypes = false;
  for (auto &param : params_decl) {
    auto paramVar = param.second;
    auto paramName = param.first;
    auto type = paramVar->getType().getCanonicalType();
    std::string typeString =
        helper::stripRefs(type).getUnqualifiedType().getAsString();
    hasFwdTypes = hasFwdTypes || type->isRValueReferenceType();
    ss << helper::getParamDeclAsString(typeString, paramName);
    if (emitRRHooks)
      ss << "init_param(" << paramName << ", " << paramName << ");\n";
  }
  // Then, with initializers...
  for (auto &param : params_expr) {
    ss << helper::getParamDeclAsString("auto", param.first, param.second);
    // No call for init_param for params with recorded init.
  }

  // Build function call
  auto parent = body->getParent();
  if (parent && parent->isRecord()) {
    // We can also get class name from qualified string...
    ss << static_cast<clang::CXXRecordDecl const *>(parent)
              ->getQualifiedNameAsString()
       << "* caller;\n";
    ss << "caller->";
  }
  ss << body->getQualifiedNameAsString();
  helper::buildExplicitTempSpec(body, ss);

  if (cudaKernel)
    ss << "<<<grid, block>>>";

  helper::buildCallExpr(body->parameters(), ss);
  ss << ";\n";

  if (emitRRHooks)
    ss << "verify_rr();\n";

  ss << "}\n";

  // include the right header for std::forward
  if (hasFwdTypes)
    output = "#include <utility>\n" + output;

  /// FIXME: Expensive string function, need to avoid this somehow!!
  if (cudaKernel)
    helper::makeCUDACompatible(output);
}

void VisitManager::pullPrimaryFnContext() {
  auto primaryDecl = primaryFn.getDefinition()->getAsFunction();
  auto parent = primaryDecl->getParent();
  auto isMemberFn = parent ? parent->isRecord() : false;

  MatchVisitor mv(*this, db);

  mv.VisitParams(primaryDecl, isMemberFn);
  registerParameterPrologue(mv, &primaryFn);
  mv.VisitTemplateParams(primaryDecl);

  auto incFile = helper::locToIncFile(primaryDecl->getLocation(),
                                      primaryDecl->getASTContext());
  if (!helper::isIncludeExternal(incFile, db)) {
    // Register includes from point of instantiation as well
    if (primaryDecl->isFunctionTemplateSpecialization())
      registerInclude(
          helper::locToIncFile(primaryDecl->getPointOfInstantiation(),
                               primaryDecl->getASTContext()));
    mv.TraverseStmt(primaryDecl->getBody());
    registerDecl(primaryDecl);
    registerInclude(incFile);
  } else {
    // if external, include all external includes
    // This will be eventually cleaned up by clang-tidy
    db.includes.getAllExternals(includes);
  }
  markVisited(primaryFn.keyName, &primaryFn);
}