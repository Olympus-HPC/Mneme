#pragma once

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeRecord.hpp"
#include "mneme/MnemeUtils.hpp"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

namespace mneme {

using namespace llvm;
using namespace llvm::object;

static void serializeFatbin(const FatBinaryWrapper_t *wrapper, FILE *out) {

  if (wrapper->Version == 1) {
    LOG_DEBUG("Writting version 1");
    auto *hdr = reinterpret_cast<const FatBinaryHeader *>(wrapper->Binary);

    std::fwrite(wrapper->Binary, hdr->HeaderSize, /* header bytes */
                1, out);
    std::fwrite(reinterpret_cast<const uint8_t *>(wrapper->Binary) +
                    hdr->HeaderSize,
                hdr->FatSize, /* payload bytes */
                1, out);
  }

  if (wrapper->Version == 2) {
    LOG_DEBUG("Writting version 2");
    const FatBinaryHeader **ptrs =
        reinterpret_cast<const FatBinaryHeader **>(wrapper->PrelinkedFatBins);

    // walk until the sentinel (magic mismatch)
    for (size_t i = 0; ptrs[i] != nullptr; ++i) {
      LOG_DEBUG("Magic is {} with version {}", ptrs[i]->Magic,
                ptrs[i]->Version);
      size_t blockSize = ptrs[i]->HeaderSize + ptrs[i]->FatSize;
      fwrite(ptrs[i], blockSize, 1, out);
      // serializeFatbin(sub + i, out);
    }
  }
}

// from cudaFatBinary.h in gpu-virtmem
typedef struct __cudaFatCudaBinary2HeaderRec {
  unsigned int magic;        // 0x466243b1 or variant
  unsigned int version;      // usually 1 or 2
  unsigned long long length; // size of the rest of this region
} __cudaFatCudaBinary2Header;

class MnemeRecorderCUDA
    : public MnemeRecorder<MnemeRecorderCUDA, mneme::DeviceVendors::CUDA> {
public:
  void extractIR() {
    LOG_INFO("Extracting IR from images");
    constexpr char OFFLOAD_BUNDLER_MAGIC_STR[] = "__CLANG_OFFLOAD_BUNDLE__";

    llvm::LLVMContext Ctx;
    for (auto &[Handle, FatbinWrapper] : HandleToBin) {
      if (FatbinWrapper->Version == 2) {
        // This is a RDC binary that includes multiple fatbins.
        for (int i = 0; FatbinWrapper->PrelinkedFatBins[i] != nullptr; i++) {
          CUmodule CUMod;
          auto PreLinkedBinary = FatbinWrapper->PrelinkedFatBins[i];
          LOG_DEBUG(
              "Prelinked fatbinary {} on address {} has version {} Magic {} ",
              i, (void *)PreLinkedBinary, PreLinkedBinary->Version,
              PreLinkedBinary->Magic);
          auto EC = DeviceTraits<DeviceVendors::CUDA>::DeviceErrorCheck(
              (cuModuleLoadData(&CUMod, PreLinkedBinary)));
          if (EC)
            LOG_FATAL("Could not load binary fatbin");
          CUdeviceptr DevPtr;
          size_t Bytes;
          EC = DeviceTraits<DeviceVendors::CUDA>::DeviceErrorCheck(
              cuModuleGetGlobal(&DevPtr, &Bytes, CUMod,
                                "_mneme_bitcode_4f_ed6118"));
          LOG_DEBUG("Found bitcode at addr {} with size {}", (void *)DevPtr,
                    Bytes);
          SmallString<4096> DeviceBitcode;
          DeviceBitcode.reserve(Bytes);
          EC = DeviceTraits<DeviceVendors::CUDA>::DeviceErrorCheck(
              cuMemcpyDtoH(DeviceBitcode.data(), DevPtr, Bytes));
          SMDiagnostic Err;
          auto M =
              parseIR(MemoryBufferRef(StringRef(DeviceBitcode.data(), Bytes),
                                      "__mneme_bitcode"),
                      Err, Ctx);
        }
      }
      CUmodule CUMod;
      auto EC = DeviceTraits<DeviceVendors::CUDA>::DeviceErrorCheck(
          (cuModuleLoadData(&CUMod, FatbinWrapper->Binary)));
      if (EC)
        LOG_FATAL("Could not load binary fatbin");
      CUdeviceptr DevPtr;
      size_t Bytes;
      EC = DeviceTraits<DeviceVendors::CUDA>::DeviceErrorCheck(
          cuModuleGetGlobal(&DevPtr, &Bytes, CUMod, "__mneme_bitcode"));
      LOG_DEBUG("Found bitcode at addr {} with size {}", (void *)DevPtr, Bytes);
      SmallString<4096> DeviceBitcode;
      DeviceBitcode.reserve(Bytes);
      EC = DeviceTraits<DeviceVendors::CUDA>::DeviceErrorCheck(
          cuMemcpyDtoH(DeviceBitcode.data(), DevPtr, Bytes));
      SMDiagnostic Err;
      auto M = parseIR(MemoryBufferRef(StringRef(DeviceBitcode.data(), Bytes),
                                       "__mneme_bitcode"),
                       Err, Ctx);
    }
  }

  void initializeGlobal(GlobalVarInfo &GVar) {
    // hipErrCheck(hipGetSymbolAddress(&GVar.DevAddr, GVar.HostSymbolAddr));
  };

  MnemeRecorderCUDA(MnemeRecorderCUDA &) = delete;
  MnemeRecorderCUDA(MnemeRecorderCUDA &&) = delete;

  MnemeRecorderCUDA() = default;
};

} // namespace mneme

template llvm::raw_ostream &
mneme::operator<<(llvm::raw_ostream &,
                  const mneme::MnemeMemoryBlob<DeviceVendors::CUDA> &);
