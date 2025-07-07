//=============================================================================
// Part of the Mneme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DESCRIPTION:
//    Find functions annotated with "jit" plus input arguments that are
//    amenable to runtime constant propagation. Stores the IR for those
//    functions, replaces them with a stub function that calls the jit runtime
//    library to compile the IR and call the function pointer of the jit'ed
//    version.
//
// USAGE:
//    1. Legacy PM
//      opt -enable-new-pm=0 -load libRRPass.dylib -legacy-rr-pass
//      -disable-output `\`
//        <input-llvm-file>
//    2. New PM
//      opt -load-pass-plugin=libRRPass.dylib -passes="rr-pass" `\`
//        -disable-output <input-llvm-file>
//
//
//===----------------------------------------------------------------------===//
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Object/ELF.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/StripDeadPrototypes.h"
#include "llvm/Transforms/IPO/StripSymbols.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <filesystem>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/FileSystem/UniqueID.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/MemoryBufferRef.h>

#include <iostream>
#include <string>

#define DEBUG_TYPE "jitpass"
#ifdef MNEME_ENABLE_DEBUG
#define DEBUG(x) x
constexpr auto debug_build = true;
#else
constexpr auto debug_build = false;
#define DEBUG(x)
#endif

#define LOG_FATAL(x)                                                           \
  report_fatal_error(llvm::Twine(std::string{} + __FILE__ + ":" +              \
                                 std::to_string(__LINE__) + " => " + x))

#if MNEME_ENABLE_HIP
constexpr char const *RegisterFunctionName = "__hipRegisterFunction";
constexpr char const *LaunchFunctionName = "hipLaunchKernel";
constexpr char const *RegisterVarName = "__hipRegisterVar";
constexpr char const *RegisterFatBinaryName = "__hipRegisterFatBinary";
#elif MNEME_ENABLE_CUDA
constexpr char const *RegisterFunctionName = "__cudaRegisterFunction";
constexpr char const *LaunchFunctionName = "cudaLaunchKernel";
constexpr char const *RegisterVarName = "__cudaRegisterVar";
constexpr char const *RegisterFatBinaryName = "__cudaRegisterFatBinary";
#else
constexpr char const *RegisterFunctionName = nullptr;
constexpr char const *LaunchFunctionName = nullptr;
constexpr char const *RegisterVarName = nullptr;
constexpr char const *RegisterFatBinaryName = nullptr;
#endif

using namespace llvm;

//-----------------------------------------------------------------------------
// MnemeRegisterIRPass implementation
//-----------------------------------------------------------------------------
namespace {

void dump(Module &M, StringRef device, StringRef phase) {

  if (!debug_build)
    return;

  std::filesystem::path ModulePath(M.getSourceFileName());
  std::filesystem::path filename(M.getSourceFileName());
  std::string rrBC(
      Twine(filename.filename().string() + "." + device + "." + phase + ".bc")
          .str());
  std::error_code EC;
  raw_fd_ostream OutBC(rrBC, EC);
  if (EC)
    throw std::runtime_error("Cannot open device code " + rrBC);
  OutBC << M;
  OutBC.close();
}

class MnemePassImpl {
private:
  Type *PtrTy = nullptr;
  Type *VoidTy = nullptr;
  Type *Int8Ty = nullptr;
  Type *Int32Ty = nullptr;
  Type *Int64Ty = nullptr;
  Type *Int128Ty = nullptr;

public:
  MnemePassImpl(Module &M) {
    PtrTy = PointerType::getUnqual(M.getContext());
    VoidTy = Type::getVoidTy(M.getContext());
    Int8Ty = Type::getInt8Ty(M.getContext());
    Int32Ty = Type::getInt32Ty(M.getContext());
    Int64Ty = Type::getInt64Ty(M.getContext());
    Int128Ty = Type::getInt128Ty(M.getContext());
  }

  bool isDeviceCompilation(Module &M) {
    Triple TargetTriple(M.getTargetTriple());
    if (TargetTriple.isNVPTX() || TargetTriple.isAMDGCN())
      return true;

    return false;
  }

  bool run(Module &M, bool IsLTO) {
    // ==================
    // Device compilation
    // ==================
    if (isDeviceCompilation(M)) {
      dump(M, "device", IsLTO ? "lto-before-mneme" : "before-mneme");
      emitJitModuleDevice(M, IsLTO);
      dump(M, "device", IsLTO ? "lto-before-mneme" : "after-mneme");
      return true;
    }

    // ================
    // Host compilation
    // ================
    dump(M, "host", IsLTO ? "lto-before-mneme" : "before-mneme");

    instrumentRegisterLinkedBinary(M);
    instrumentRegisterFatBinary(M);
    instrumentRegisterFatBinaryEnd(M);
    instrumentRegisterVar(M);
    instrumentRegisterFunction(M);

    if (verifyModule(M, &errs()))
      LOG_FATAL("Broken original module found, compilation aborted!");

    dump(M, "host", IsLTO ? "lto-before-mneme" : "after-mneme");

    return true;
  }

private:
  std::string getJitBitcodeUniqueName(Module &M) {
    llvm::sys::fs::UniqueID ID;
    if (auto EC = llvm::sys::fs::getUniqueID(M.getSourceFileName(), ID))
      LOG_FATAL("Could not get unique id");

    SmallString<64> Out;
    llvm::raw_svector_ostream OutStr(Out);
    OutStr << "_mneme_bitcode" << llvm::format("_%x", ID.getDevice())
           << llvm::format("_%x", ID.getFile());

    return std::string(Out);
  }

  void emitJitModuleDevice(Module &M, bool IsLTO) {
    std::string BitcodeStr;
    raw_string_ostream OS(BitcodeStr);
    WriteBitcodeToFile(M, OS);

    std::string GVName =
        (IsLTO ? "__jit_bitcode_lto" : getJitBitcodeUniqueName(M));
    //  NOTE: HIP compilation supports custom section in the binary to store
    //  the IR. CUDA does not, hence we parse the IR by reading the global
    //  from the device memory.
    Constant *DeviceModule = ConstantDataArray::get(
        M.getContext(), ArrayRef<uint8_t>((const uint8_t *)BitcodeStr.data(),
                                          BitcodeStr.size()));
    auto *GV =
        new GlobalVariable(M, DeviceModule->getType(), /* isConstant */ true,
                           GlobalValue::ExternalLinkage, DeviceModule, GVName);
    appendToUsed(M, {GV});
    GV->setSection(".jit.bitcode" + (IsLTO ? ".lto" : getUniqueModuleId(&M)));
  }

  /* HOST INSTRUMENTATION */

  Function *getOrCreateFunction(Module &M, FunctionType *FnTy,
                                StringRef FnName) {
    LLVMContext &Ctx = M.getContext();
    Function *F = M.getFunction(FnName);
    if (!F) {
      F = Function::Create(FnTy, GlobalValue::ExternalLinkage, FnName, M);
      F->addFnAttr(Attribute::NoInline);
      F->setDSOLocal(true);

      // Create dummy body that does nothing
      BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
      IRBuilder<> B(BB);
      B.CreateRetVoid();
    }
    return F;
  }

  FunctionCallee getOrCreateRegisterLinkedBinaryFn(Module &M) {
    FunctionType *JitRegisterLinkedBinaryFnTy =
        FunctionType::get(VoidTy, {PtrTy, PtrTy},
                          /* isVarArg=*/false);
    return M.getOrInsertFunction("__register_linked_binary",
                                 JitRegisterLinkedBinaryFnTy);
  }

  FunctionCallee getOrCreateRegisterFatbinary(Module &M) {
    LLVMContext &Ctx = M.getContext();
    FunctionType *RegisterFatbinaryFnTy =
        FunctionType::get(VoidTy, {PtrTy, PtrTy, PtrTy},
                          /* isVarArg=*/false);
    return M.getOrInsertFunction("__register_fatbinary", RegisterFatbinaryFnTy);
  }

  FunctionCallee getOrCreateRegisterFatBinaryEndFn(Module &M) {
    FunctionType *RegisterFatBinaryEndFnTy =
        FunctionType::get(VoidTy, {PtrTy},
                          /* isVarArg=*/false);

    return M.getOrInsertFunction("__register_fatbinary_end",
                                 RegisterFatBinaryEndFnTy);
  }

  FunctionCallee getOrCreateRegisterVarFn(Module &M) {
    FunctionType *RegisterVarFnTy = FunctionType::get(
        VoidTy,
        {PtrTy, PtrTy, PtrTy, PtrTy, Int32Ty, Int64Ty, Int32Ty, Int32Ty},
        /* isVarArg=*/false);
    return M.getOrInsertFunction("__register_var", RegisterVarFnTy);
  }

  FunctionCallee getOrCreateRegisterFunctionFn(Module &M) {
    FunctionType *RegisterFunctionFnTy =
        FunctionType::get(VoidTy,
                          {PtrTy, PtrTy, PtrTy, PtrTy, Int32Ty, PtrTy, PtrTy,
                           PtrTy, PtrTy, PtrTy},
                          /* isVarArg=*/false);
    return M.getOrInsertFunction("__register_function", RegisterFunctionFnTy);
  }

  void instrumentRegisterLinkedBinary(Module &M) {
// This is CUDA specific.
#if !MNEME_ENABLE_CUDA
    return;
#endif

    // Note: we check for __cuda_fatibn_wrapper to avoid emitting for the
    // link.stub. It's not strictly necessary since this module will not have a
    // device bitcode to pull and we skip at runtime.
    if (!M.getGlobalVariable("__cuda_fatbin_wrapper", /*AllowInternal=*/true)) {
      return;
    }

    FunctionCallee JitRegisterLinkedBinaryFn =
        getOrCreateRegisterLinkedBinaryFn(M);

    for (auto &F : M.getFunctionList()) {
      if (!F.getName().starts_with("__cudaRegisterLinkedBinary"))
        continue;

      for (auto *User : F.users()) {
        CallBase *CB = dyn_cast<CallBase>(User);
        if (!CB)
          continue;

        IRBuilder<> Builder(CB);
        std::string GVName = getJitBitcodeUniqueName(M);
        auto *Arg = Builder.CreateGlobalString(GVName);
        Builder.CreateCall(JitRegisterLinkedBinaryFn,
                           {CB->getArgOperand(1), Arg});
      }
    }
  }

  void instrumentRegisterFatBinary(Module &M) {
    Function *F = nullptr;

    if (!RegisterFatBinaryName)
      return;

    F = M.getFunction(RegisterFatBinaryName);
    if (!F)
      return;

    FunctionCallee JitRegisterFatBinaryFn = getOrCreateRegisterFatbinary(M);

    for (auto *User : F->users()) {
      CallBase *CB = dyn_cast<CallBase>(User);
      if (!CB)
        continue;

      IRBuilder<> Builder(CB->getNextNode());
      Value *FatbinWrapper = CB->getArgOperand(0);

      std::string GVName = getJitBitcodeUniqueName(M);
      auto *Arg = Builder.CreateGlobalString(GVName);

      Builder.CreateCall(JitRegisterFatBinaryFn, {CB, FatbinWrapper, Arg});
    }
  }

  void instrumentRegisterFatBinaryEnd(Module &M) {
// This is CUDA specific.
#if !MNEME_ENABLE_CUDA
    return;
#endif

    Function *F = M.getFunction("__cudaRegisterFatBinaryEnd");
    if (!F)
      return;

    FunctionCallee JitRegisterFatBinaryEndFn =
        getOrCreateRegisterFatBinaryEndFn(M);

    for (auto *User : F->users()) {
      CallBase *CB = dyn_cast<CallBase>(User);
      if (!CB)
        continue;

      IRBuilder<> Builder(CB->getNextNode());
      Value *FatbinWrapper = CB->getArgOperand(0);
      Builder.CreateCall(JitRegisterFatBinaryEndFn, {FatbinWrapper});
    }
  }

  void instrumentRegisterVar(Module &M) {
    Function *RegisterVarFn = nullptr;
    if (!RegisterVarName)
      return;

    RegisterVarFn = M.getFunction(RegisterVarName);
    if (!RegisterVarFn)
      return;

    FunctionCallee JitRegisterVarFn = getOrCreateRegisterVarFn(M);

    for (User *Usr : RegisterVarFn->users())
      if (CallBase *CB = dyn_cast<CallBase>(Usr)) {
        IRBuilder<> Builder(CB->getNextNode());
        SmallVector<Value *> Args{CB->args()};
        Builder.CreateCall(JitRegisterVarFn, Args);
      }
  }

  void instrumentRegisterFunction(Module &M) {
    Function *RegisterFunctionFn = nullptr;
    if (!RegisterFunctionName)
      return;

    RegisterFunctionFn = M.getFunction(RegisterFunctionName);
    if (!RegisterFunctionFn)
      return;

    FunctionCallee JitRegisterFunctionFn = getOrCreateRegisterFunctionFn(M);

    for (User *Usr : RegisterFunctionFn->users())
      if (CallBase *CB = dyn_cast<CallBase>(Usr)) {
        IRBuilder<> Builder(CB->getNextNode());
        SmallVector<Value *> Args{CB->args()};
        Builder.CreateCall(JitRegisterFunctionFn, Args);
      }
  }
};

// New PM implementation.
struct MnemePass : PassInfoMixin<MnemePass> {
  MnemePass(bool IsLTO) : IsLTO(IsLTO) {}
  bool IsLTO;

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    MnemePassImpl Mneme{M};

    bool Changed = Mneme.run(M, IsLTO);
    if (Changed)
      return PreservedAnalyses::none();

    return PreservedAnalyses::all();
  }

  // Without isRequired returning true, this pass will be skipped for
  // functions decorated with the optnone LLVM attribute. Note that clang -O0
  // decorates all functions with optnone.
  static bool isRequired() { return true; }
};

// Legacy PM implementation.
struct LegacyMnemePass : public ModulePass {
  static char ID;
  LegacyMnemePass() : ModulePass(ID) {}
  bool runOnModule(Module &M) override {
    MnemePassImpl Mneme{M};
    bool Changed = Mneme.run(M, false);
    return Changed;
  }
};
} // namespace

//-----------------------------------------------------------------------------
// New PM Registration
//-----------------------------------------------------------------------------
llvm::PassPluginLibraryInfo getMnemePassPluginInfo() {
  const auto Callback = [](PassBuilder &PB) {
    // TODO: decide where to insert it in the pipeline. Early avoids
    // inlining jit function (which disables jit'ing) but may require more
    // optimization, hence overhead, at runtime. We choose after early
    // simplifications which should avoid inlining and present a reasonably
    // analyzable IR module.

    // NOTE: For device jitting it should be possible to register the pass late
    // to reduce compilation time and does lose the kernel due to inlining.
    // However, there are linking errors, working assumption is that the hiprtc
    // linker cannot re-link already linked device libraries and aborts.

    PB.registerPipelineStartEPCallback(
        // PB.registerOptimizerLastEPCallback(
        // PB.registerPipelineEarlySimplificationEPCallback(
        [&](ModulePassManager &MPM, auto) {
          MPM.addPass(MnemePass{false});
          return true;
        });

    PB.registerFullLinkTimeOptimizationEarlyEPCallback(
        [&](ModulePassManager &MPM, auto) {
          MPM.addPass(MnemePass{true});
          return true;
        });
  };

  return {LLVM_PLUGIN_API_VERSION, "MnemePass", LLVM_VERSION_STRING, Callback};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getMnemePassPluginInfo();
}

//-----------------------------------------------------------------------------
// Legacy PM Registration
//-----------------------------------------------------------------------------
// The address of this variable is used to uniquely identify the pass. The
// actual value doesn't matter.
char LegacyMnemePass::ID = 0;

static RegisterPass<LegacyMnemePass>
    X("legacy-mneme-pass", "Mneme Pass",
      false, // This pass doesn't modify the CFG => false
      false  // This pass is not a pure analysis pass => false
    );
