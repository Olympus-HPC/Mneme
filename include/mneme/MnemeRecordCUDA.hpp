#pragma once

#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeRecord.hpp"
#include "mneme/MnemeUtils.hpp"

namespace mneme {

class MnemeRecorderCUDA : public MnemeRecorder<MnemeRecorderCUDA, CUDA> {
public:
  void extractIR() {
    LOG_INFO("Extracting IR from images");
    constexpr char OFFLOAD_BUNDLER_MAGIC_STR[] = "__CLANG_OFFLOAD_BUNDLE__";

    for (auto &[Handle, FatbinWrapper] : HandleToBin) {
      size_t Pos = 0;
      LOG_DEBUG("Processing Handle and FatbinWrapper {} {}", (void *)Handle,
                (void *)FatbinWrapper);
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
