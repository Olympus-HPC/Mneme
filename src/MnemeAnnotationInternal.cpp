#include "mneme/MnemeAnnotation.hpp"
#include "mneme/MnemeUtils.hpp"
#include <llvm/Support/raw_ostream.h>

namespace mneme {
namespace metadata {
void serialize(llvm::raw_ostream &OS, const Metadata &Md) {
  OS.write(reinterpret_cast<const char *>(&Md.builtin),
           sizeof(std::underlying_type_t<BuiltinDType>));
  OS.write(reinterpret_cast<const char *>(&Md.threshold), sizeof(double));
  OS.write(reinterpret_cast<const char *>(&Md.threshold_kind),
           sizeof(ThresholdKind));
  OS.write(reinterpret_cast<const char *>(&Md.norm), sizeof(Norm));
  size_t Size = 0;
  if (Md.tag.has_value())
    Size = Md.tag->size();

  OS.write(reinterpret_cast<const char *>(&Size), sizeof(size_t));
  if (Md.tag.has_value())
    OS.write(reinterpret_cast<const char *>(Md.tag->c_str()), Size);
}

size_t serializedSize(const Metadata &Md) {
  return sizeof(std::underlying_type_t<BuiltinDType>) + sizeof(double) +
         sizeof(ThresholdKind) + sizeof(Norm) + sizeof(size_t) +
         (Md.tag ? Md.tag->size() : 0);
}

Metadata fromBuffer(const char *&Buffer) {
  Metadata Md;
  Md.builtin = util::extractScalar<BuiltinDType>(Buffer);
  Md.threshold = util::extractScalar<double>(Buffer);
  Md.threshold_kind = util::extractScalar<ThresholdKind>(Buffer);
  Md.norm = util::extractScalar<Norm>(Buffer);
  size_t Size = util::extractScalar<size_t>(Buffer);
  if (Size) {
    Md.tag = std::string(Buffer, Size);
    Buffer += Size;
  }
  return Md;
}

} // namespace metadata
} // namespace mneme
