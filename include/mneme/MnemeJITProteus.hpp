#include <proteus/CoreLLVM.h>
#include <proteus/CoreLLVMDevice.h>

#include <llvm/Support/InitLLVM.h>
#ifdef MNEME_ENABLE_HIP
#include <proteus/CoreLLVMHIP.h>
#elif defined(MNEME_ENABLE_CUDA)
#else
#error "Please define MNEME_ENABLE_HIP or MNEME_ENABLE_CUDA"
#endif

namespace mneme {

void pruneMnemeGlobals(llvm::Module &M) {
  using namespace llvm;
  SmallVector<GlobalVariable *> GlobalsToErase;
  for (auto &GV : M.globals()) {
    auto Name = GV.getName();
    if (Name.starts_with("_mneme_bitcode")) {
      GlobalsToErase.push_back(&GV);
      removeFromUsedLists(M, [&GV](Constant *C) {
        if (auto *Global = dyn_cast<GlobalVariable>(C))
          return Global == &GV;
        return false;
      });
    }
  }
  for (auto GV : GlobalsToErase) {
    M.eraseGlobalVariable(GV);
  }
}
} // namespace mneme
