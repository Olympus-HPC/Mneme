// ZeroInitAllocas.cpp
#include "../llvm/core.h"

#include <mneme/MnemeLogger.hpp>

#include "llvm/IR/Attributes.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace {

static bool hasAutoInitAttr(const CallInst *CI) {
  // Match string attribute "auto-init" (value optional, e.g., "zero")
  return CI->getAttributes()
           .hasAttributeAtIndex(AttributeList::FunctionIndex, "auto-init");
}

struct RemoveAutoInitMemsetsTransform {
  static bool run(Module &M) {
    SmallVector<CallInst*, 32> ToErase;

    for (Function &F : M) {
      if (F.isDeclaration()) continue;

      for (BasicBlock &BB : F)
        for (Instruction &I : BB)
          if (auto *CI = dyn_cast<CallInst>(&I)) {
            if (auto *II = dyn_cast<IntrinsicInst>(CI)) {
              if (II->getIntrinsicID() == Intrinsic::memset &&
                  hasAutoInitAttr(CI)) {
                ToErase.push_back(CI);
              }
            }
          }
    }

    for (CallInst *CI : ToErase)
      CI->eraseFromParent();

    return !ToErase.empty();
  }
};

} // namespace

extern "C" {
API_EXPORT(void)
TransformPy_RemoveAutoInitMemset(LLVMModuleRef Mod) {
  auto *M = unwrap(Mod);
  bool Modified = RemoveAutoInitMemsetsTransform::run(*M);
  LOG_DEBUG("RemoveAutoInitMemsets {} modify LLVM IR",
            Modified ? "did" : "did not");
}
}
