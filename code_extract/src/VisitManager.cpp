#include "VisitManager.h"
#include "Visitor.h"

#include <string>
#include <vector>

#include "clang/AST/Attrs.inc"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
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
  if (includePath == "" || includePath.find(".c") != std::string::npos)
    return false;

  // Obviously there is no requirement for there to be an include folder in the
  // path but for now we assume there is for simplicity. We can also change this
  // to match against the last '/' instead.
  std::string incFile;
  if (includePath.find(".h") == std::string::npos) {
    // For potential forwarding headers, split at last '/'
    incFile = includePath.substr(includePath.rfind("/") + 1);
  } else
    incFile = includePath.substr(includePath.rfind("include") + 7 + 1);

  // If the include start with _, they probably are from builtins so ignore.
  if (incFile[0] == '_')
    return false;

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
    // If expression is lambda, save that...
    if (type->hasUnnamedOrLocalType()) {
      // Could possibly be lambda object,
      // find underlying expression.
      auto recordDecl = type->getAsCXXRecordDecl();

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
      params_expr.push_back({key, lmbExpr});

      fillParams(key, lmbExpr->capture_begin(), lmbExpr->capture_end());
    } else {
      expr->getNameAsString();
      params_decl.push_back({key, expr});
    }
    /// FIXME: We can do more here, like also handling function pointers like we
    /// do lambdas...
  }
}

void VisitManager::registerParameterPrologue(ObjInfo *fnObj) {
  auto specDecl = fnObj->getSpecialization();
  clang::FunctionDecl *fnDecl = nullptr;
  if (specDecl)
    fnDecl = specDecl->getAsFunction();
  else
    fnDecl = fnObj->getDefiniton()->getAsFunction();

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
    ss << "\"" << primaryFn.getKeyName() // Function key names will always be their mangled names...
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
  for (auto &param : params_decl) {
    auto paramVar = param.second;
    auto paramName = param.first;
    std::string typeString =
        paramVar->getType().getCanonicalType().getAsString();
    ss << helper::getParamDeclAsString(typeString, paramName);
    if (emitRRHooks)
      ss << "init_param(" << paramName << ", " << paramName << ");\n";
  }
  // Then, with initializers...
  for (auto &param : params_expr) {
    std::string init;
    llvm::raw_string_ostream stream(init);
    param.second->printPretty(stream, nullptr, body->getLangOpts());
    ss << helper::getParamDeclAsString("auto", param.first, init);
    // No call for init_param for params with recorded init.
  }

  // Build function call
  ss << body->getQualifiedNameAsString();
  if (cudaKernel)
    ss << "<<<grid, block>>>";

  auto numParams = body->getNumParams();
  int paramCount = 0;
  ss << "(";
  if (paramCount < numParams)
    ss << "p" << paramCount;
  for (paramCount++; paramCount < numParams; paramCount++)
    ss << ", " << "p" << paramCount;
  ss << ");\n";

  if (emitRRHooks)
    ss << "verify_rr();\n";

  ss << "}\n";
}

void VisitManager::pullPrimaryFnContext() {
  auto primaryDecl = primaryFn.getDefiniton()->getAsFunction();

  MatchVisitor mv(*this, db);
  mv.VisitParams(primaryDecl);
  registerParameterPrologue(&primaryFn);

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