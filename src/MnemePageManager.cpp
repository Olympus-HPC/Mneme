#include "MnemePageManager.hpp"
#include "Utils.hpp"
#include <cstdint>

using namespace mneme;

ContiguousAddrBlock::ContiguousAddrBlock(uintptr_t start, uint64_t sz)
    : PageAddr(start), Size(sz) {}

// Comparison operators for sorting blocks by address and size
bool ContiguousAddrBlock::operator<(const ContiguousAddrBlock &other) const {
  return PageAddr < other.PageAddr;
}

// Function to coalesce contiguous blocks
void PageManager::coalesce() {
  if (FreeVARanges.size() < 2)
    return; // Nothing to coalesce if less than two blocks

  auto it = FreeVARanges.begin();
  while (it != FreeVARanges.end()) {
    auto next_it = std::next(it);

    if (next_it == FreeVARanges.end())
      break;

    // Check if the current block is contiguous with the next block
    if (it->PageAddr + it->Size == next_it->PageAddr) {
      // Merge the two blocks
      ContiguousAddrBlock mergedBlock = {it->PageAddr,
                                         it->Size + next_it->Size};

      // Remove the original two blocks
      it = FreeVARanges.erase(it);
      next_it = FreeVARanges.erase(next_it);

      // Insert the merged block
      FreeVARanges.insert(mergedBlock);

      // Start over from the merged block (to ensure we handle multiple
      // contiguous blocks)
      it = FreeVARanges.find(mergedBlock);
    } else {
      ++it;
    }
  }
}

// Find a block that has a size >= requestedSize
std::multiset<ContiguousAddrBlock>::iterator
PageManager::findFreeBlock(size_t requestedSize) {
  for (auto it = FreeVARanges.begin(); it != FreeVARanges.end(); ++it) {
    auto size = it->Size;
    if (size >= requestedSize) {
      return it;
    }
  }
  return FreeVARanges.end();
}

std::pair<uintptr_t, uint64_t>
PageManager::reserveBestFitPage(uint64_t VASize) {
  // We need to always reserve at least a single page
  uint64_t ReqSize = util::roundUp(VASize, PageSize);
  auto FreeNode = findFreeBlock(ReqSize);
  if (FreeNode == FreeVARanges.end())
    return std::make_pair((uintptr_t) nullptr, 0);

  auto Ptr = FreeNode->PageAddr;
  auto NodePageSize = FreeNode->Size;

  FreeVARanges.erase(FreeNode);

  if (ReqSize == NodePageSize)
    return std::make_pair(Ptr, ReqSize);

  auto NewPtr = Ptr + ReqSize;
  auto RemainingSize = NodePageSize - ReqSize;

  ContiguousAddrBlock block{NewPtr, RemainingSize};
  FreeVARanges.insert(block);

  // This can be expensive. Currently we coalesce in every request that
  // modifies our free-pages.
  coalesce();

  return std::make_pair(Ptr, ReqSize);
}

std::multiset<ContiguousAddrBlock>::iterator
PageManager::findInclusivePage(uintptr_t Addr, size_t Size) {
  uintptr_t request_end = Addr + Size;
  for (auto it = FreeVARanges.begin(); it != FreeVARanges.end(); ++it) {
    uintptr_t block_end = it->PageAddr + it->Size;
    if (it->PageAddr <= Addr && block_end >= request_end) {
      return it;
    }
  }
  return FreeVARanges.end();
}

std::pair<uintptr_t, uint64_t> PageManager::requestExactPage(uint64_t VASize,
                                                             void *VA) {
  uint64_t ReqSize = util::roundUp(VASize, PageSize);
  auto FreeNode = findInclusivePage((uintptr_t)VA, ReqSize);
  if (FreeNode == FreeVARanges.end())
    return std::make_pair((uintptr_t) nullptr, 0);

  auto Ptr = FreeNode->PageAddr;
  auto NodePageSize = FreeNode->Size;

  FreeVARanges.erase(FreeNode);

  // We found exactly the requested page.
  if (ReqSize == NodePageSize && (uintptr_t)VA == Ptr)
    return std::make_pair(Ptr, ReqSize);

  if (VA != nullptr) {
    if ((uintptr_t)VA < Ptr ||
        ((uintptr_t)VA + VASize) > (Ptr + NodePageSize)) {
      std::ostringstream oss;
      oss << "Unable to return requested address: " << std::hex
          << reinterpret_cast<uintptr_t>(VA)
          << " instead the returned address is " << std::hex
          << reinterpret_cast<uintptr_t>(Ptr) << std::dec << "\n";
      FATAL_ERROR(oss.str());
    }
  }

  // There are 'unused' addresses left from the requested one
  // We add them back to the page manager
  auto NewNodePageSize = (uintptr_t)VA - Ptr;
  ContiguousAddrBlock block{Ptr, NewNodePageSize};
  FreeVARanges.insert(block);

  // There are 'unused' addresses right/higher than the end of
  // the requested page addresses
  auto NewPtr = (uintptr_t)VA + ReqSize;
  if (NewPtr < Ptr + NodePageSize) {
    ContiguousAddrBlock block{NewPtr, Ptr + NodePageSize - NewPtr};
    FreeVARanges.insert(block);
  }
  // This can be expensive. Currently we coalesce in every request that
  // modifies our free-pages.
  coalesce();

  return std::make_pair((uintptr_t)VA, ReqSize);
}

std::pair<uintptr_t, uint64_t> PageManager::allocateAddr(uint64_t VASize,
                                                         void *VA) {
  if (VA == nullptr)
    return reserveBestFitPage(VASize);
  return requestExactPage(VASize, VA);
}

PageManager::PageManager(uint64_t VASize, uint64_t PageSize, void *VA,
                         int32_t DeviceID)
    : TotalVASize(VASize), ReservedVA(reinterpret_cast<uintptr_t>(VA)),
      PageSize(PageSize), DeviceID(DeviceID) {
  FreeVARanges.insert(ContiguousAddrBlock{ReservedVA, TotalVASize});
}

void PageManager::releaseAddr(uint64_t VASize, void *VA) {
  ContiguousAddrBlock AddrBlock{reinterpret_cast<uint64_t>(VA), VASize};
  FreeVARanges.insert(AddrBlock);
  coalesce();
}
