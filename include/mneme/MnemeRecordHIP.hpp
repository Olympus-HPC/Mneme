#pragma once

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeMemoryHIP.hpp"
#include "mneme/MnemeRecord.hpp"
#include "mneme/MnemeUtils.hpp"

#include <dlfcn.h>
#include <hip/hip_runtime.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StableHashing.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Object/ELF.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>

namespace mneme {

class MnemeRecorderHIP : public MnemeRecorder<MnemeRecorderHIP, HIP> {
public:
  static auto *getRTLib() { return dlopen("libamdhip64.so", RTLD_NOW); }
  static constexpr const char *getLaunchKernelFnName() {
    return "hipLaunchKernel";
  }
  static constexpr const char *getDeviceMallocFnName() { return "hipMalloc"; }
  static constexpr const char *getPinnedMallocFnName() {
    return "hipHostMalloc";
  }
  static constexpr const char *getManagedMallocFnName() {
    return "hipMallocManaged";
  }
  static constexpr const char *getDeviceFreeFnName() { return "hipFree"; }
  static constexpr const char *getPinnedFreeFnName() { return "hipHostFree"; }
  static constexpr const char *getUURegisterFunctionFnName() {
    return "__hipRegisterFunction";
  }
  static const char *getUURegisterVarFnName() { return "__hipRegisterVar"; }
  static const char *getUURegisterFatbinFnName() {
    return "__hipRegisterFatBinary";
  }

  static constexpr bool hasFatBinEnd = false;

  const std::string getArch() {
    static std::string Arch{[]() {
      int Device;
      hipErrCheck(hipInit(0));
      hipErrCheck(hipGetDevice(&Device));
      hipDeviceProp_t DeviceProperties;

      // Get properties of the current device
      hipErrCheck(hipGetDeviceProperties(&DeviceProperties, Device));

      // Get the full architecture name (e.g., gfx90a:sramecc+:xnack-)
      std::string arch_name = DeviceProperties.gcnArchName;

      LOG_DEBUG("Detected architecture {}", arch_name);
      // Find the colon (:) to isolate the base architecture
      auto DevArch = std::string(arch_name.substr(0, arch_name.find(':')));
      LOG_DEBUG("Detected Device Architecture is {}", DevArch);
      return DevArch;
    }()};

    return Arch;
  }

  void extractIR() {
    LOG_INFO("Extracting IR from images");
    constexpr char OFFLOAD_BUNDLER_MAGIC_STR[] = "__CLANG_OFFLOAD_BUNDLE__";

    for (auto &[Handle, FatbinWrapper] : HandleToBin) {
      size_t Pos = 0;
      LOG_DEBUG("Processing Handle and FatbinWrapper {} {}", (void *)Handle,
                (void *)FatbinWrapper);
      const char *Binary = FatbinWrapper->Binary;
      llvm::StringRef Magic(Binary, sizeof(OFFLOAD_BUNDLER_MAGIC_STR) - 1);
      if (Magic != OFFLOAD_BUNDLER_MAGIC_STR)
        LOG_FATAL("Error missing magic string");
      Pos += sizeof(OFFLOAD_BUNDLER_MAGIC_STR) - 1;

      auto Read8ByteIntLE = [](const char *S, size_t Pos) {
        return llvm::support::endian::read64le(S + Pos);
      };

      uint64_t NumberOfBundles = Read8ByteIntLE(Binary, Pos);
      if (NumberOfBundles == 0) {
        LOG_CRITICAL("Number of Bundles is 0");
        continue;
      }
      LOG_DEBUG("Processing number of bundles: {}", NumberOfBundles);
      Pos += 8;

      llvm::StringRef DeviceBinary;
      for (uint64_t i = 0; i < NumberOfBundles; ++i) {
        uint64_t Offset = Read8ByteIntLE(Binary, Pos);
        Pos += 8;

        uint64_t Size = Read8ByteIntLE(Binary, Pos);
        Pos += 8;

        uint64_t TripleSize = Read8ByteIntLE(Binary, Pos);
        Pos += 8;

        llvm::StringRef Triple(Binary + Pos, TripleSize);
        Pos += TripleSize;

        if (!Triple.contains("amdgcn") || !Triple.contains(getArch()))
          continue;
        LOG_DEBUG("Processing bundle {}", Triple.str());

        DeviceBinary = llvm::StringRef(Binary + Offset, Size);
      }

      auto DeviceElf = llvm::object::ELF64LEFile::create(DeviceBinary);
      if (DeviceElf.takeError())
        LOG_FATAL("Cannot create the device elf");

      auto Sections = DeviceElf->sections();
      if (Sections.takeError())
        LOG_FATAL("Error reading sections");

      llvm::ArrayRef<uint8_t> DeviceBitcode;

      llvm::LLVMContext Ctx;

      auto extractModuleFromSection = [&DeviceElf,
                                       &Ctx](auto &Section,
                                             llvm::StringRef SectionName) {
        llvm::ArrayRef<uint8_t> BitcodeData;
        auto SectionContents = DeviceElf->getSectionContents(Section);
        if (SectionContents.takeError())
          LOG_FATAL("Error reading section contents");
        BitcodeData = *SectionContents;
        auto Bitcode =
            llvm::StringRef{reinterpret_cast<const char *>(BitcodeData.data()),
                            BitcodeData.size()};

        llvm::SMDiagnostic Err;
        auto M = parseIR(llvm::MemoryBufferRef{Bitcode, SectionName}, Err, Ctx);
        if (!M)
          LOG_FATAL("unexpected");
        return M;
      };

      // We extract bitcode from sections. If there is a .jit.bitcode.lto
      // section due to RDC compilation that's the only bitcode we need,
      // othewise we collect all .jit.bitcode sections.

      llvm::SmallVector<std::unique_ptr<llvm::Module>> LLVMModules;
      for (auto Section : *Sections) {
        auto SectionName = DeviceElf->getSectionName(Section);
        if (SectionName.takeError())
          LOG_FATAL("Error reading section name");

        if (!SectionName->starts_with(".jit.bitcode"))
          continue;

        auto M = extractModuleFromSection(Section, *SectionName);
        LOG_DEBUG("Processing section with name {}", SectionName.get().str());

        if (*SectionName == ".jit.bitcode.lto") {
          LLVMModules.clear();
          LLVMModules.push_back(std::move(M));
          LOG_DEBUG("Found LTO module");
          break;
        } else {
          LLVMModules.push_back(std::move(M));
        }
      }

      llvm::DenseMap<std::string, llvm::Function *> KernelNameToFunction;
      for (auto &Mod : LLVMModules) {
        for (llvm::Function &Func : *Mod.get()) {
          // Skip non kernels
          if (Func.getCallingConv() != llvm::CallingConv::AMDGPU_KERNEL)
            continue;

          // Can a declarion have a calling conv, if no this is unecessary.
          if (Func.isDeclaration())
            continue;

          KernelNameToFunction[Func.getName().str()] = &Func;
        }
      }

      auto getFuncDescr = [&, this](llvm::Function &F) {
        llvm::SmallVector<uint64_t, 8> RRInfo;
        auto DL = F.getParent()->getDataLayout();
        for (auto &A : F.args()) {
          // Datatypes such as structs passed by value to kernels are copied
          // into a parameter vector. Over here we test whether an argument is
          // byval, if it is we know on the host side this invocation forwards
          // the arguments by value
          if (A.hasByRefAttr() || A.hasByValAttr()) {
            RRInfo.emplace_back(
                DL.getTypeStoreSize(A.getPointeeInMemoryValueType()));
          } else {
            RRInfo.emplace_back(DL.getTypeStoreSize(A.getType()));
          }
        }
        return RRInfo;
      };

      auto getArgNames = [&, this](llvm::Function &F) {
        llvm::SmallVector<std::string> RRInfo;
        for (auto &A : F.args()) {
          RRInfo.emplace_back(A.getName().str());
        }
        return RRInfo;
      };

      auto canSpecialize = [&, this](llvm::Function &F) {
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
      };

      auto convertToDouble = [&, this](llvm::Function &F) {
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
      };

      auto &CurrKernels = HandleToKernels[Handle];
      for (auto &KI : CurrKernels) {
        LOG_DEBUG("Searching for kernel with name {}", KI->getName());
        auto Iter = KernelNameToFunction.find(KI->getName());
        if (Iter == KernelNameToFunction.end()) {
          BlackList.insert(KI->getFunHandle());
          continue;
        }

        llvm::Function *KFunc = Iter->second;
        KI->setArgSizes(getFuncDescr(*KFunc));
        KI->setArgNames(getArgNames(*KFunc));
        KI->setSpecializations(canSpecialize(*KFunc));
        KI->setToDoubleFunc(convertToDouble(*KFunc));
      }

      for (auto &Mod : LLVMModules) {
        auto [StableHash, FName] = storeModule(*Mod);
        for (auto &KI : CurrKernels) {
          KI->ModuleFiles.push_back(FName);
          KI->updateHash(StableHash);
        }
      }
    }
  }

  void initializeGlobal(GlobalVarInfo &GVar) {
    hipErrCheck(hipGetSymbolAddress(&GVar.DevAddr, GVar.HostSymbolAddr));
  };

  MnemeRecorderHIP(MnemeRecorderHIP &) = delete;
  MnemeRecorderHIP(MnemeRecorderHIP &&) = delete;

  MnemeRecorderHIP() = default;
};

} // namespace mneme

template llvm::raw_ostream &
mneme::operator<<(llvm::raw_ostream &,
                  const mneme::MnemeMemoryBlob<DeviceVendors::HIP> &);
