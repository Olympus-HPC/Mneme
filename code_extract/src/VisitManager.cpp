#include "VisitManager.h"
#include "Visitor.h"

#include <string>
#include <unordered_map>
#include <vector>

#include "clang/AST/Attrs.inc"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/LLVM.h"
#include "llvm/Support/raw_ostream.h"

namespace helper {

class LambdaCallback : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  clang::LambdaExpr const *lambdaExpr = nullptr;

  virtual void
  run(const clang::ast_matchers::MatchFinder::MatchResult &Result) final {
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
  auto prefixEnd = std::min(typeString.find_last_of(')'), typeString.size());
  std::string prefix = typeString.substr(0, prefixEnd);
  std::string suffix = typeString.substr(prefixEnd);
  std::string decl = prefix + " " + name + suffix;
  if (!init.empty())
    decl += " = " + init;
  return decl + ";\n";
}

void buildCallExpr(clang::ArrayRef<clang::ParmVarDecl *> const &parameters,
                   llvm::raw_string_ostream &ss,
                   std::string paramPrefix = "p") {
  int paramCount = 0;
  ss << "( ";
  for (auto param : parameters) {
    std::string paramName = paramPrefix + std::to_string(paramCount);
    auto type = param->getType();
    if (type->isRValueReferenceType()) {
      auto strippedType =
          helper::stripRefs(type).getUnqualifiedType().getAsString();
      ss << "std::forward<" << strippedType << ">(" << paramName << ")";
    } else
      ss << paramName;
    ss << ",";
    paramCount++;
  }
  ss.str().back() = ')';
}

/// FIXME: Move into generic utils namespace
extern clang::QualType getUnderlyingType(clang::QualType const &type);
} // namespace helper

void VisitManager::registerDecl(clang::NamedDecl const *decl) {
  declRefs.push_back(decl);
}

void VisitManager::registerDecl(clang::TagDecl const *decl) {
  tagDecls.push_back(decl);
}

void VisitManager::registerDecl(clang::TypedefNameDecl const *decl) {
  tagDecls.push_back(decl);
}

void VisitManager::registerDecl(clang::FunctionDecl const *decl) {
  // Do not emit inlined functions
  if (decl->isCXXClassMember() && decl->isInlined())
    return;
  if (auto tmpDecl = decl->getPrimaryTemplate())
    declRefs.push_back(tmpDecl);
  else
    declRefs.push_back(decl);
}

bool VisitManager::registerInclude(std::string const &includePath) {
  // We will likely not pick up any source files here so no need to check.
  if (includePath == "")
    return false;

  // However, we might find other types of implementation files we need to ignore.
  /// FIXME: Make a set of unacceptable file extensions...
  if (includePath.find(".tcc") != std::string::npos)
    return true;

  // Obviously there is no requirement for there to be an include folder in the
  // path but for now we assume there is for simplicity. We can also change this
  // to match against the last '/' instead.
  std::string fileName = includePath.substr(includePath.rfind("/") + 1);
  std::string incFile;
  if (fileName.find(".") == std::string::npos) {
    // For potential forwarding headers, split at last '/'
    incFile = fileName;
  } else
    incFile = includePath.substr(includePath.rfind("include") + 7 + 1);

  // If the include start with _, they probably are from builtins so ignore.
  if (incFile[0] == '_')
    return true;

  includes.insert(incFile);
  return true;
}

bool VisitManager::isVisited(std::string const &name) {
  return visitedNodes.find(name) != visitedNodes.end();
}

ObjInfo const *VisitManager::getVisitedObj(std::string const &name) {
  return visitedNodes.at(name);
}

void VisitManager::markVisited(std::string name, ObjInfo const *objInfo) {
  visitedNodes.insert({name, objInfo});
}

void VisitManager::addToVisit(clang::Stmt *stmt) { toVisitNodes.push(stmt); }

template <typename T>
void VisitManager::fillParams(std::string const &prefix, T *begin, T *end) {
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

      addToVisit(lmbExpr->getBody());
      std::string lmbBody;
      llvm::raw_string_ostream stream(lmbBody);
      lmbExpr->printPretty(stream, nullptr, recordDecl->getLangOpts());
      params_expr.push_back({key, lmbBody});

      fillParams(key, lmbExpr->capture_begin(), lmbExpr->capture_end());
    } else if (recordDecl) {
      clang::CXXConstructorDecl *ctor = nullptr;
      for (auto constructor : recordDecl->ctors()) {
        if (constructor->isDeleted() || constructor->isCopyOrMoveConstructor())
          continue;
        ctor = constructor;
      }
      if (!ctor || !ctor->getNumParams()) {
        params_decl.push_back({key, expr});
        continue;
      }

      std::string ctorCall;
      llvm::raw_string_ostream stream(ctorCall);
      stream << recordDecl->getQualifiedNameAsString();
      helper::buildCallExpr(ctor->parameters(), stream, key + "_");
      params_expr.push_back({key, ctorCall});

      fillParams(key.substr(1) + "_", ctor->param_begin(), ctor->param_end());
    } else {

      params_decl.push_back({key, expr});
    }
    /// FIXME: We can do more here, like also handling function pointers like we
    /// do lambdas...
  }
}

void VisitManager::registerParameterPrologue(ObjInfo *fnObj) {
  clang::FunctionDecl *fnDecl = fnObj->getDefiniton()->getAsFunction();

  auto numParams = fnDecl->getNumParams();
  if (!numParams)
    return;

  fillParams("", fnDecl->param_begin(), fnDecl->param_end());
}

void VisitManager::emitStandaloneFile(std::string &output, bool emitRR,
                                      std::string const &configString) {
  auto body =
      static_cast<clang::FunctionDecl const *>(primaryFn.getDefiniton());
  bool cudaKernel = body->hasAttr<clang::CUDAGlobalAttr>();
  bool emitRRHooks = cudaKernel && emitRR;

  llvm::raw_string_ostream ss(output);

  for (auto &inc : includes) {
    ss << "#include ";
    if (inc.find('.') == std::string::npos)
      ss << "<" << inc << ">";
    else
      ss << "\"" << inc << "\"";
    ss << "\n\n";
  }
  if (emitRRHooks)
    ss << "#include \"RRHooks.h\"\n\n";

  for (auto &tags : tagDecls) {
    // Same with tags, add missing semicolon!
    tags->print(ss);
    ss << ";\n\n";
  }

  // Emit all declrefs (functions calls + global refs)
  for (auto ref_it = declRefs.rbegin(); ref_it != declRefs.rend(); ref_it++) {
    auto decl = *ref_it;
    decl->print(ss);
    if (decl->getKind() == clang::Decl::Kind::Var)
      ss << ";";
    ss << "\n\n";
  }

  // Building main
  ss << "int main(int argc, char *argv[]) {\n";

  if (emitRRHooks) {
    // First add the prologue
    ss << "init_RR(";
    ss << "\"" << primaryFn.getKeyName() // Function key names will always be
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
  auto isMemberFn = parent ? parent->isRecord() : false;
  if (isMemberFn) {
    // We can also get class name from qualified string...
    ss << static_cast<clang::CXXRecordDecl const *>(parent)
              ->getQualifiedNameAsString()
       << "* caller;\n";
    ss << "caller->";
  }
  ss << body->getQualifiedNameAsString();
  if (auto tmpSpec = body->getTemplateSpecializationInfo()) {
    auto tmpArgs = tmpSpec->TemplateArguments->asArray();
    ss << "< ";
    for (auto arg : tmpArgs) {
      if (arg.getKind() != clang::TemplateArgument::ArgKind::Type)
        continue;
      auto argType = arg.getAsType();
      if (argType->hasUnnamedOrLocalType())
        break;
      ss << argType.getAsString() << ",";
    }
    ss.str().back() = '>';
  }

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
}

void VisitManager::pullPrimaryFnContext() {
  auto primaryDecl = primaryFn.getDefiniton()->getAsFunction();
  auto parent = primaryDecl->getParent();
  auto isMemberFn = parent ? parent->isRecord() : false;

  MatchVisitor mv(*this, db);

  mv.VisitParams(primaryDecl, isMemberFn);
  registerParameterPrologue(&primaryFn);
  mv.VisitTemplateParams(primaryDecl);

  auto extSource = primaryFn.getExtSourceFile();
  if (extSource.empty()) {
    addToVisit(primaryDecl->getBody());
    registerDecl(primaryDecl);
  } else {
    registerInclude(extSource);
  }
  markVisited(primaryFn.getKeyName(), &primaryFn);

  while (!toVisitNodes.empty()) {
    auto stmt = toVisitNodes.front();
    toVisitNodes.pop();
    mv.TraverseStmt(stmt);
  }
  /// TODO: Complete
}