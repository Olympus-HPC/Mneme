#include "VisitManager.h"
#include "Visitor.h"

#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "clang/AST/Attrs.inc"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/Mangle.h"
#include "llvm/Support/raw_ostream.h"

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

void VisitManager::getParamInstantiationsAsString(
    clang::FunctionDecl const *fnDecl, std::vector<std::string> &params) {
  auto numParams = fnDecl->getNumParams();
  if (!numParams)
    return;

  // Typically we would want to find all function call to this method
  // but for now we can just use types from functionDecl.
  // Right now we only build the declarations, eventually we should be able to
  // recreate the values...
  /// TODO: this
  for (int i = 0; i < numParams; i++) {
    auto expr = fnDecl->parameters()[i];
    std::string typeString = expr->getType().getCanonicalType().getAsString();
    auto prefixEnd = std::min(typeString.find_last_of(')'), typeString.size());
    std::string prefix = typeString.substr(0, prefixEnd);
    std::string suffix = typeString.substr(prefixEnd);
    params.push_back(prefix + " p" + std::to_string(i) + suffix + ";\n");
  }
}

void VisitManager::emitStandaloneFile(std::string &output,
                                      std::string const &configString) {
  auto body =
      static_cast<clang::FunctionDecl const *>(primaryFn.getDefiniton());
  bool cudaKernel = body->hasAttr<clang::CUDAGlobalAttr>();

  llvm::raw_string_ostream ss(output);

  for (auto &inc : includes) {
    ss << "#include ";
    if (inc.find('.') == std::string::npos)
      ss << "<" << inc << ">";
    else
      ss << "\"" << inc << "\"";
    ss << "\n";
  }
  if (cudaKernel) 
    ss << "#include \"RRHooks.h\"\n";
  ss << '\n';

  for (auto &tags : tagDecls) {
    // Same with tags, add missing semicolon!
    tags->print(ss);
    ss << ";\n";
  }

  // Emit all declrefs (functions calls + global refs)
  for (auto ref_it = declRefs.rbegin(); ref_it != declRefs.rend(); ref_it++) {
    auto decl = *ref_it;
    decl->print(ss);
    if (decl->getKind() == clang::Decl::Kind::Var)
      ss << ";";
    ss << "\n";
  }

  // Building main
  ss << "int main(int argc, char *argv[]) {\n";

  if (cudaKernel) {
    // First add the prologue
    ss << "init_RR(";
    ss << "\"" << clang::ASTNameGenerator(body->getASTContext()).getName(body) << "\"";
    ss << ", argc, argv);\n";

    // If we have a cuda kernel, we should add the config string...
    ss << "dim3 grid;\n"
       << "dim3 block;\n";

    // ...and load it from RR
    ss << "init_dims(\"Grid\", grid);\n"
       << "init_dims(\"Block\", block);\n";
  }

  std::vector<std::string> paramVec;
  getParamInstantiationsAsString(body, paramVec);
  for (int i = 0; i < paramVec.size(); i++) {
    ss << paramVec[i];
    if (cudaKernel)
      ss << "init_param(" << i << ", p" << i << ");\n";
  }

  // Build function call
  ss << body->getNameAsString();
  if (cudaKernel)
    ss << "<<<grid, block>>>";
  auto numParams = body->getNumParams();
  int paramCount = 0;
  ss << "(";
  for (; paramCount < numParams - 1; paramCount++)
    ss << "p" << paramCount << ", ";

  if (paramCount < numParams)
    ss << "p" << paramCount;
  ss << ");\n";

  if (cudaKernel) 
    ss << "verify_rr();\n";

  ss << "}\n";
}

void VisitManager::pullPrimaryFnContext() {
  auto primaryDecl = primaryFn.getDefiniton()->getAsFunction();
  addToVisit(primaryDecl->getBody());
  registerDecl(primaryDecl);
  markVisited(primaryFn.getName(), &primaryFn);

  MatchVisitor mv(*this, db);
  mv.VisitParms(primaryDecl);
  while (!toVisitNodes.empty()) {
    auto stmt = toVisitNodes.front();
    toVisitNodes.pop();
    mv.TraverseStmt(stmt);
  }
  /// TODO: Complete
}