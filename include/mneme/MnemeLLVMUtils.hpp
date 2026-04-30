#pragma once
#include "llvm/ADT/DenseMapInfo.h"
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/Twine.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>

#include "mneme/MnemeKernelInfo.hpp"
#include "mneme/MnemeLogger.hpp"

namespace llvm {
template <> struct DenseMapInfo<std::string> {
  static inline std::string getEmptyKey() {
    return std::string(); // Define an empty key
  }
  static inline std::string getTombstoneKey() {
    return std::string("<TOMBSTONE_KEY>");
  }
  static unsigned getHashValue(const std::string &Key) {
    // Use std::hash for hashing the string
    return std::hash<std::string>{}(Key);
  }
  static bool isEqual(const std::string &LHS, const std::string &RHS) {
    return LHS == RHS;
  }
};
} // namespace llvm
//
namespace mneme {
inline static void collectPointerOffsets(llvm::Type *Ty,
                                         const llvm::DataLayout &DL,
                                         uint64_t BaseOffset,
                                         llvm::SmallVectorImpl<size_t> &Out) {
  if (Ty->isPointerTy()) {
    Out.emplace_back(BaseOffset);
    return;
  }

  if (auto *STy = llvm::dyn_cast<llvm::StructType>(Ty)) {
    if (STy->isOpaque())
      return;

    const auto *SL = DL.getStructLayout(STy);
    for (unsigned I = 0; I < STy->getNumElements(); ++I)
      collectPointerOffsets(STy->getElementType(I), DL,
                            BaseOffset + SL->getElementOffset(I), Out);
    return;
  }

  if (auto *ATy = llvm::dyn_cast<llvm::ArrayType>(Ty)) {
    uint64_t Stride = DL.getTypeAllocSize(ATy->getElementType());
    for (uint64_t I = 0; I < ATy->getNumElements(); ++I)
      collectPointerOffsets(ATy->getElementType(), DL, BaseOffset + I * Stride,
                            Out);
    return;
  }

  if (auto *VTy = llvm::dyn_cast<llvm::FixedVectorType>(Ty)) {
    uint64_t Stride = DL.getTypeAllocSize(VTy->getElementType());
    for (unsigned I = 0; I < VTy->getNumElements(); ++I)
      collectPointerOffsets(VTy->getElementType(), DL, BaseOffset + I * Stride,
                            Out);
  }
}

inline static llvm::SmallVector<llvm::SmallVector<size_t>>
getPointerOffsetsByArg(llvm::Function &F) {
  llvm::SmallVector<llvm::SmallVector<size_t>> PointerOffsets;
  const auto &DL = F.getParent()->getDataLayout();
  for (auto &A : F.args()) {
    llvm::SmallVector<size_t> ArgOffsets;
    llvm::Type *ArgStorageTy = A.getType();
    if (A.hasByRefAttr() || A.hasByValAttr())
      ArgStorageTy = A.getPointeeInMemoryValueType();
    collectPointerOffsets(ArgStorageTy, DL, 0, ArgOffsets);
    PointerOffsets.emplace_back(std::move(ArgOffsets));
  }
  return PointerOffsets;
}

inline static llvm::SmallVector<KernelArgPointerSlot>
getKernelPointerSlots(llvm::Function &F) {
  llvm::SmallVector<KernelArgPointerSlot> PointerSlots;
  auto ArgPointerOffsets = getPointerOffsetsByArg(F);
  for (size_t ArgIndex = 0; ArgIndex < ArgPointerOffsets.size(); ++ArgIndex)
    for (auto Offset : ArgPointerOffsets[ArgIndex])
      PointerSlots.push_back(KernelArgPointerSlot{ArgIndex, Offset});
  return PointerSlots;
}

inline static llvm::SmallVector<std::string> getArgNames(llvm::Function &F) {
  llvm::SmallVector<std::string> Info;
  for (auto &A : F.args()) {
    auto name = A.getName().str();
    if (name.size() > 0)
      Info.push_back(name);
    else
      Info.push_back("");
  }
  return Info;
}

inline static llvm::SmallVector<bool> canSpecialize(llvm::Function &F) {
  llvm::SmallVector<bool> RRInfo;
  for (auto &A : F.args()) {
    llvm::Type *ArgType = A.getType();
    if (ArgType->isIntegerTy(1)) {
      RRInfo.emplace_back(true);
    } else if (ArgType->isIntegerTy(8)) {
      RRInfo.emplace_back(true);
    } else if (ArgType->isIntegerTy(32)) {
      RRInfo.emplace_back(true);
    } else if (ArgType->isIntegerTy(64)) {
      RRInfo.emplace_back(true);
    } else if (ArgType->isFloatTy()) {
      RRInfo.emplace_back(true);
    } else if (ArgType->isDoubleTy()) {
      RRInfo.emplace_back(true);
    } else if (ArgType->isX86_FP80Ty() || ArgType->isPPC_FP128Ty() ||
               ArgType->isFP128Ty()) {
      RRInfo.emplace_back(true);
    } else {
      RRInfo.emplace_back(false);
    }
  }
  return RRInfo;
}

inline static llvm::SmallVector<std::function<double(void *)>>
convertToDouble(llvm::Function &F) {
  llvm::SmallVector<std::function<double(void *)>> RRInfo;
  for (auto &A : F.args()) {
    llvm::Type *ArgType = A.getType();
    if (ArgType->isIntegerTy(1)) {
      RRInfo.emplace_back([](void *data) {
        bool val = *(bool *)data;
        return (double)val;
      });
    } else if (ArgType->isIntegerTy(8)) {
      RRInfo.emplace_back([](void *data) {
        int8_t val = *(int8_t *)data;
        return (double)val;
      });
    } else if (ArgType->isIntegerTy(32)) {
      RRInfo.emplace_back([](void *data) {
        int32_t val = *(int32_t *)data;
        return (double)val;
      });
    } else if (ArgType->isIntegerTy(64)) {
      RRInfo.emplace_back([](void *data) {
        int64_t val = *(int64_t *)data;
        return (double)val;
      });
    } else if (ArgType->isFloatTy()) {
      RRInfo.emplace_back([](void *data) {
        float val = *(float *)data;
        return (double)val;
      });
    } else if (ArgType->isDoubleTy()) {
      RRInfo.emplace_back([](void *data) {
        double val = *(double *)data;
        return (double)val;
      });
    } else if (ArgType->isX86_FP80Ty() || ArgType->isPPC_FP128Ty() ||
               ArgType->isFP128Ty()) {
      RRInfo.emplace_back([](void *data) {
        double val = *(double *)data;
        return (double)val;
      });
    } else {
      RRInfo.emplace_back([](void *data) {
        double val = *(double *)data;
        return -1.0;
      });
    }
  }
  return RRInfo;
}

inline static llvm::SmallVector<uint64_t, 8> getFuncDescr(llvm::Function &F) {
  llvm::SmallVector<uint64_t, 8> RRInfo;
  auto DL = F.getParent()->getDataLayout();
  for (auto &A : F.args()) {
    // Datatypes such as structs passed by value to kernels are copied
    // into a parameter vector. Over here we test whether an argument is
    // byval, if it is we know on the host side this invocation forwards
    // the arguments by value
    if (A.hasByRefAttr() || A.hasByValAttr()) {
      RRInfo.emplace_back(DL.getTypeStoreSize(A.getPointeeInMemoryValueType()));
    } else {
      RRInfo.emplace_back(DL.getTypeStoreSize(A.getType()));
    }
  }
  return RRInfo;
}
} // namespace mneme
