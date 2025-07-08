#pragma once
#include <filesystem>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeSymbols.hpp"

namespace mneme {

using DeviceHandle = void **;

struct FatBinaryHeader {
  unsigned int Magic;         // == FATBIN_MAGIC
  unsigned short Version;     // == FATBIN_VERSION
  unsigned short HeaderSize;  // size of this header (bytes, multiple of 8)
  unsigned long long FatSize; // size of the remainder of the fat-binary
};

struct FatBinaryWrapper_t {
  int Magic;
  int Version;
  const char *Binary;
  const FatBinaryHeader **PrelinkedFatBins;
};

struct MnemeDeviceLinkedBin {
  /* This is provided by the Device Driver and we are agnostic to it */
  const DeviceHandle Handle;
  /* The binary corrensponds to this linked fat binary */
  const FatBinaryWrapper_t *FatBinary;
  bool ExtractedCode;
  llvm::SmallVector<llvm::StringRef> ModuleIds;
  llvm::SmallVector<std::unique_ptr<llvm::Module>> Modules;
  llvm::SmallVector<llvm::StringRef> BinaryBlobs;
  llvm::SmallVector<std::string> ModuleFiles;
  llvm::SmallVector<std::string> BinaryFiles;
  llvm::DenseMap<llvm::StringRef, llvm::Function *> KernelNameToFunction;
  llvm::stable_hash LinkedBinHash;

  MnemeDeviceLinkedBin(DeviceHandle Handle, FatBinaryWrapper_t *Wrapper)
      : Handle(Handle), FatBinary(Wrapper), ExtractedCode(false) {}

  MnemeDeviceLinkedBin()
      : Handle(nullptr), FatBinary(nullptr), ExtractedCode(false) {}

  template <DeviceVendors Vendor> bool ExtractCode(llvm::LLVMContext &Ctx) {
    if (ExtractedCode)
      return false;
    ExtractedCode = true;
    LOG_DEBUG("Going to extract code {}", (void *)this);
    if (FatBinary->Version == 1)
      ExtractCodeWithoutRDC<Vendor>(Ctx, Modules);
    else if (FatBinary->Version == 2)
      ExtractCodeWithRDC<Vendor>(Ctx, Modules, BinaryBlobs);
    else
      LOG_FATAL("Unknown Fatbinary format");

    return true;
  }

  template <DeviceVendors Vendor>
  void StoreModules(std::string &RecordReplayDir) {
    LinkedBinHash = 0;
    for (auto &M : Modules) {
      std::string StrBuffer;
      llvm::raw_string_ostream RSO(StrBuffer);

      // Serialize the module into the string buffer
      llvm::WriteBitcodeToFile(*M, RSO);
      RSO.flush();
      auto [Filename, Hash] =
          storeBlob(RecordReplayDir, StrBuffer, "RecoredIR_", ".bc");
      ModuleFiles.push_back(Filename);
      LinkedBinHash = llvm::stable_hash_combine(LinkedBinHash, Hash);
    }

    for (auto &B : BinaryBlobs) {
      auto [Filename, Hash] =
          storeBlob(RecordReplayDir, B, "RecoredBlob_", ".bin");
      BinaryFiles.push_back(Filename);
      LinkedBinHash = llvm::stable_hash_combine(LinkedBinHash, Hash);
    }
  }

  template <DeviceVendors Vendor> void FindKernels() {
    for (auto &M : Modules) {
      FindKernels<Vendor>(KernelNameToFunction, *M);
    }
  }

  llvm::stable_hash getStaticHash() const { return LinkedBinHash; }

  std::optional<llvm::Function *> getKernelFunction(llvm::StringRef FuncName) {
    for (auto &KV : KernelNameToFunction) {
      LOG_DEBUG("DB contains {} {}", KV.first.str(),
                KV.second->getName().str());
    }
    auto Func = KernelNameToFunction.find(FuncName);
    if (Func == KernelNameToFunction.end())
      return std::nullopt;
    return Func->second;
  }

  const llvm::ArrayRef<std::string> getModuleIRFiles() const {
    return ModuleFiles;
  }

  const llvm::ArrayRef<std::string> getModuleBinFiles() const {
    return BinaryFiles;
  }

private:
  template <mneme::DeviceVendors Vendor>
  void ExtractCodeWithoutRDC(
      llvm::LLVMContext &Ctx,
      llvm::SmallVector<std::unique_ptr<llvm::Module>> &Modules);

  template <mneme::DeviceVendors Vendor>
  void
  ExtractCodeWithRDC(llvm::LLVMContext &Ctx,
                     llvm::SmallVector<std::unique_ptr<llvm::Module>> &Modules,
                     llvm::SmallVector<llvm::StringRef> &Blobs);

  template <mneme::DeviceVendors Vendor>
  void FindKernels(
      llvm::DenseMap<llvm::StringRef, llvm::Function *> &KernelNameToFunction,
      llvm::Module &M);

  std::pair<std::string, llvm::hash_code>
  storeBlob(std::string &RecordReplayDir, llvm::StringRef Buffer,
            llvm::StringRef Prefix, llvm::StringRef Suffix) {
    std::error_code EC;

    uint64_t StableHash = llvm::stable_hash_combine_string(
        llvm::StringRef(Buffer.data(), Buffer.size()));

    std::string Filename(
        std::filesystem::path(llvm::Twine(RecordReplayDir + "/" + Prefix +
                                          std::to_string(StableHash) + Suffix)
                                  .str())
            .string());

    llvm::raw_fd_ostream OutBC(Filename, EC);
    if (EC)
      LOG_FATAL("Cannot write module ir file");

    OutBC << Buffer;
    LOG_DEBUG("Stored Blob with StaticHash:{} to file {}", StableHash,
              std::filesystem::canonical(Filename).string());
    OutBC.close();
    return std::make_pair(Filename, StableHash);
  }
};
} // namespace mneme
