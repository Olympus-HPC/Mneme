#pragma once
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <set>
#include <sys/types.h>

#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeUtils.hpp"

struct ContiguousAddrBlock {
  // Starting address of the free block
  uintptr_t PageAddr;
  // Size of the free block
  uint64_t Size;

  ContiguousAddrBlock(uintptr_t start, uint64_t sz);

  // Comparison operators for sorting blocks by address and size
  bool operator<(const ContiguousAddrBlock &other) const;
};

class PageManager {

protected:
  std::set<ContiguousAddrBlock> FreeVARanges;
  uintptr_t ReservedVA;
  uint64_t TotalVASize;
  uint64_t PageSize;
  int32_t DeviceID;

  // Erase a block from the set
  bool EraseVirtualAddress(uintptr_t addr, size_t size);
  // Function to coalesce contiguous blocks
  void coalesce();

  // Find a block that has a size >= requestedSize
  std::set<ContiguousAddrBlock>::iterator findFreeBlock(size_t requestedSize);
  // Find a block that includes the range [Addr, Addr + size)

  std::set<ContiguousAddrBlock>::iterator findInclusivePage(uintptr_t Addr,
                                                            size_t Size);

  std::pair<uintptr_t, uint64_t> reserveBestFitPage(uint64_t VASize);

  std::pair<uintptr_t, uint64_t> requestExactPage(uint64_t VASize, void *VA);

public:
  PageManager() = default;
  PageManager(uint64_t VASize, uint64_t PageSize, void *VA, int32_t DeviceID);

  std::pair<uintptr_t, uint64_t> allocateAddr(uint64_t VASize, void *VA);

  void releaseAddr(uint64_t VASize, void *VA);

  void *getVAStart() const {
    LOG_DEBUG("Returning address {}", ReservedVA);
    return reinterpret_cast<void *>(ReservedVA);
  }
  uint64_t getTotalVASize() const { return TotalVASize; }
  uint64_t getNumPages() const { return FreeVARanges.size(); }
  uint64_t getUniqueNumPages() const {
    std::set<ContiguousAddrBlock> s(FreeVARanges.begin(), FreeVARanges.end());
    return s.size();
  }
  void dump() const {
    for (auto V : FreeVARanges) {
      LOG_INFO("Page Manager Start: {} Size {}", (void *)V.PageAddr, V.Size);
    }
  }
};

template <typename MnemeDeviceRT>
std::unique_ptr<PageManager> initializePageManager(int DeviceID,
                                                   void *ReqAddr = nullptr,
                                                   uint64_t ActualSize = -1) {
  const int MaxTries = 1;
  auto MinPageSize = MnemeDeviceRT::getMinPageSize(DeviceID);
  if (ActualSize == -1)
    ActualSize =
        mneme::util::roundUp(MnemeDeviceRT::getFixedMemorySize(), MinPageSize);
  void *VA = nullptr;
  int Try = 0;
  if (!ReqAddr)
    ReqAddr = reinterpret_cast<void *>(MnemeDeviceRT::getSuggestedAddr());

  while (VA != ReqAddr && Try < MaxTries) {
    LOG_INFO("Trying {}/{} to Reserve Virtual Address {} space of size {}...",
             Try, MaxTries, reinterpret_cast<void *>(ReqAddr), ActualSize);

    if (VA)
      MnemeDeviceRT::freeVirtualAddress(VA, ActualSize);

    VA = MnemeDeviceRT::getVirtualAddress(
        ActualSize, reinterpret_cast<void *>(ReqAddr), MinPageSize);
    Try++;
  }
  LOG_INFO("... Reserved successfully Virtual Address {}", VA);
  return std::make_unique<PageManager>(ActualSize, MinPageSize, VA, 0);
}
