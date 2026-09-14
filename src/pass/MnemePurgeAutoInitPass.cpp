#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>
#if __has_include(<llvm/Plugins/PassPlugin.h>)
#include <llvm/Plugins/PassPlugin.h>
#elif __has_include(<llvm/Passes/PassPlugin.h>)
#include <llvm/Passes/PassPlugin.h>
#else
#error "Cannot find LLVM PassPlugin.h"
#endif

using namespace llvm;

namespace {

// Clang tags the initialization it emits for -ftrivial-auto-var-init with
// !annotation !{!"auto-init"}. Annotation metadata holds the strings directly
// or, when annotations are grouped, tuples of strings; both are accepted by
// the IR verifier, so check for either shape.
bool hasAutoInitAnnotation(const Instruction &I) {
  const MDNode *Annotations = I.getMetadata(LLVMContext::MD_annotation);
  if (!Annotations)
    return false;

  auto IsAutoInit = [](const Metadata *MD) {
    const auto *Str = dyn_cast_or_null<MDString>(MD);
    return Str && Str->getString() == "auto-init";
  };

  for (const MDOperand &Op : Annotations->operands()) {
    if (IsAutoInit(Op.get()))
      return true;

    if (const auto *Group = dyn_cast_or_null<MDNode>(Op.get()))
      for (const MDOperand &GroupOp : Group->operands())
        if (IsAutoInit(GroupOp.get()))
          return true;
  }

  return false;
}

// Removes the llvm.memset calls Clang emits for -ftrivial-auto-var-init.
// Mneme records with forced zero-initialization; replay strips it to match
// the original application.
class PurgeAutoInitPass : public PassInfoMixin<PurgeAutoInitPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    SmallVector<CallInst *, 32> ToErase;

    for (Function &F : M) {
      if (F.isDeclaration())
        continue;

      for (BasicBlock &BB : F)
        for (Instruction &I : BB)
          if (auto *II = dyn_cast<IntrinsicInst>(&I))
            if (II->getIntrinsicID() == Intrinsic::memset &&
                hasAutoInitAnnotation(*II))
              ToErase.push_back(II);
    }

    for (CallInst *CI : ToErase)
      CI->eraseFromParent();

    return ToErase.empty() ? PreservedAnalyses::all()
                           : PreservedAnalyses::none();
  }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "MnemePurgeAutoInit", "0.1",
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name != "purge-autoinit")
                    return false;
                  MPM.addPass(PurgeAutoInitPass());
                  return true;
                });
          }};
}
