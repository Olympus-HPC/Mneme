#pragma once
#include "mneme/MnemeAnnotation.hpp"
#include <llvm/Support/raw_ostream.h>

namespace mneme{
namespace metadata{
void serialize(llvm::raw_ostream &OS, const Metadata& Md);
Metadata fromBuffer(const char *&Buffer);
}
}
