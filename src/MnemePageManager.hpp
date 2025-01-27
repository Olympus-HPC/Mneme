#pragma once
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <set>
#include <sys/types.h>

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
  std::multiset<ContiguousAddrBlock> FreeVARanges;
  uintptr_t ReservedVA;
  uint64_t TotalVASize;
  uint64_t PageSize;
  int32_t DeviceID;

  // Erase a block from the multiset
  bool EraseVirtualAddress(uintptr_t addr, size_t size);
  // Function to coalesce contiguous blocks
  void coalesce();

  // Find a block that has a size >= requestedSize
  std::multiset<ContiguousAddrBlock>::iterator
  findFreeBlock(size_t requestedSize);
  // Find a block that includes the range [Addr, Addr + size)

  std::multiset<ContiguousAddrBlock>::iterator findInclusivePage(uintptr_t Addr,
                                                                 size_t Size);

  std::pair<uintptr_t, uint64_t> reserveBestFitPage(uint64_t VASize);

  std::pair<uintptr_t, uint64_t> requestExactPage(uint64_t VASize, void *VA);

public:
  PageManager(uint64_t VASize, uint64_t PageSize, void *VA, int32_t DeviceID);

  std::pair<uintptr_t, uint64_t> allocateAddr(uint64_t VASize, void *VA);

  void releaseAddr(uint64_t VASize, void *VA);
};
