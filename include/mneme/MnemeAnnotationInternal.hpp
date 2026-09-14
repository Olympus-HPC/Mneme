#pragma once
#include "mneme/MnemeAnnotation.hpp"
#include <cstddef>
#include <llvm/Support/raw_ostream.h>

namespace mneme {
namespace metadata {
void serialize(llvm::raw_ostream &OS, const Metadata &Md);
Metadata fromBuffer(const char *&Buffer);
// The number of bytes serialize() writes for Md.
size_t serializedSize(const Metadata &Md);
} // namespace metadata
} // namespace mneme
