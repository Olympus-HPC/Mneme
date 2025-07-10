#include <llvm/IR/Function.h>
#include <llvm/Object/ELF.h>

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeDeviceBinary.hpp"

using namespace mneme;
using namespace llvm;

template <>
void MnemeDeviceLinkedBin::ExtractCodeWithoutRDC<DeviceVendors::HIP>(
    LLVMContext &Ctx, SmallVector<std::unique_ptr<Module>> &Modules) {
  constexpr char OFFLOAD_BUNDLER_MAGIC_STR[] = "__CLANG_OFFLOAD_BUNDLE__";
  if (ModuleIds.size() != 1)
    LOG_FATAL("Expected Non RDC module to contain a single LLVM IR");

  size_t Pos = 0;
  LOG_DEBUG("Processing Handle and FatbinWrapper {} {}", (void *)Handle,
            (void *)FatBinary);
  const char *Binary = FatBinary->Binary;
  llvm::StringRef Magic(Binary, sizeof(OFFLOAD_BUNDLER_MAGIC_STR) - 1);
  if (Magic != OFFLOAD_BUNDLER_MAGIC_STR)
    LOG_FATAL("Error missing magic string");
  Pos += sizeof(OFFLOAD_BUNDLER_MAGIC_STR) - 1;

  auto Read8ByteIntLE = [](const char *S, size_t Pos) {
    return llvm::support::endian::read64le(S + Pos);
  };

  uint64_t NumberOfBundles = Read8ByteIntLE(Binary, Pos);
  if (NumberOfBundles == 0)
    LOG_FATAL("Number of Bundles is 0");

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

    if (!Triple.contains("amdgcn") ||
        !Triple.contains(DeviceTraits<HIP>::GetDeviceArch()))
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

  auto extractModuleFromSection =
      [&DeviceElf, &Ctx](auto &Section, llvm::StringRef SectionName) {
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
  for (auto Section : *Sections) {
    auto SectionName = DeviceElf->getSectionName(Section);
    if (SectionName.takeError())
      LOG_FATAL("Error reading section name");

    if (!SectionName->starts_with(".jit.bitcode"))
      continue;

    auto M = extractModuleFromSection(Section, *SectionName);
    LOG_DEBUG("Processing section with name {}", SectionName.get().str());

    if (*SectionName == ".jit.bitcode.lto") {
      Modules.clear();
      Modules.push_back(std::move(M));
      LOG_DEBUG("Found LTO module");
      break;
    } else {
      Modules.push_back(std::move(M));
    }
  }
}

template <>
void MnemeDeviceLinkedBin::FindKernels<DeviceVendors::HIP>(
    llvm::DenseMap<llvm::StringRef, llvm::Function *> &KernelNameToFunction,
    llvm::Module &M) {
  for (llvm::Function &Func : M) {
    // Skip non kernels
    if (Func.getCallingConv() != llvm::CallingConv::AMDGPU_KERNEL)
      continue;

    // Can a declarion have a calling conv, if no this is unecessary.
    if (Func.isDeclaration())
      continue;
    LOG_DEBUG("Adding kernel {}", Func.getName().str());

    KernelNameToFunction.try_emplace(Func.getName(), &Func);
  }
}
