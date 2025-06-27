#pragma once
#include "llvm/ADT/DenseMapInfo.h"
#include <llvm/ADT/Twine.h>

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
