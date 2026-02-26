#pragma once
//===----------------------------------------------------------------------===//
// mneme_annotate.h - User-facing annotation API for Mneme
//
// This header declares mneme::annotate(ptr, Metadata{...}) and related types.
// It is intentionally *interface-only*: no lambdas / comparators yet.
//
// Example:
//   double* p = ...;
//   mneme::annotate(p, mneme::Metadata{
//     .dtype = mneme::Type::Builtin,
//     .builtin = mneme::BuiltinDType::F64,
//     .threshold = 0.1,
//     .threshold_kind = mneme::ThresholdKind::Absolute,
//     .norm = mneme::Norm::Linf,
//   });
//
// You can also use the typed helper:
//   mneme::annotate<double>(p, mneme::Metadata{ .threshold = 0.1 });
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <optional>
#include <string>
#include <iostream>

namespace mneme {

// Built-in scalar dtypes Mneme knows about today.
enum class BuiltinDType : std::uint8_t {
  U8 = 0,
  I8,
  U16,
  I16,
  U32,
  I32,
  U64,
  I64,
  F16,
  F32,
  F64,
};

// How to interpret `threshold`.
enum class ThresholdKind : std::uint8_t {
  Absolute = 0,
  Relative = 1,
};

// If the pointer represents a collection, which norm to use for aggregation.
enum class Norm : std::uint8_t {
  None = 0, // interpret threshold per-element / per-scalar
  L1,
  L2,
  Linf,
};

// Core metadata attached to a pointer/region.
struct Metadata {
  // Builtin dtype (used when dtype == Builtin).
  BuiltinDType builtin = BuiltinDType::U8;

  // Error threshold semantics.
  double threshold = 0.0;
  ThresholdKind threshold_kind = ThresholdKind::Absolute;

  // Aggregation semantics (for arrays/regions). Optional.
  Norm norm = Norm::None;

  // Optional user tag (for reporting / grouping). Not interpreted by Mneme.
  std::optional<std::string> tag = std::nullopt;
};

// --------- Builtin dtype mapping helpers (optional sugar) ------------------

template <class T>
struct builtin_dtype_of {
  static constexpr BuiltinDType value = BuiltinDType::U8;
};

template <> struct builtin_dtype_of<float>  { static constexpr BuiltinDType value = BuiltinDType::F32; };
template <> struct builtin_dtype_of<double> { static constexpr BuiltinDType value = BuiltinDType::F64; };

template <> struct builtin_dtype_of<std::int8_t>   { static constexpr BuiltinDType value = BuiltinDType::I8;  };
template <> struct builtin_dtype_of<std::uint8_t>  { static constexpr BuiltinDType value = BuiltinDType::U8;  };
template <> struct builtin_dtype_of<std::int16_t>  { static constexpr BuiltinDType value = BuiltinDType::I16; };
template <> struct builtin_dtype_of<std::uint16_t> { static constexpr BuiltinDType value = BuiltinDType::U16; };
template <> struct builtin_dtype_of<std::int32_t>  { static constexpr BuiltinDType value = BuiltinDType::I32; };
template <> struct builtin_dtype_of<std::uint32_t> { static constexpr BuiltinDType value = BuiltinDType::U32; };
template <> struct builtin_dtype_of<std::int64_t>  { static constexpr BuiltinDType value = BuiltinDType::I64; };
template <> struct builtin_dtype_of<std::uint64_t> { static constexpr BuiltinDType value = BuiltinDType::U64; };

// --------- Implementation hook (defined in Mneme library) -----------------

namespace detail {

// The library provides the implementation of this function.
// It must be safe to call multiple times for the same pointer (idempotent or last-wins).
void annotate_impl(const void* ptr, Metadata md);

} // namespace detail

// --------- User-facing API ------------------------------------------------

// Primary user API: annotate a pointer with metadata.
inline void annotate(const void* ptr, Metadata md) {
  detail::annotate_impl(ptr, md);
}

// Convenience overload for non-const pointers.
inline void annotate(void* ptr, Metadata md) {
  detail::annotate_impl(ptr, md);
}

// Typed helper: sets builtin dtype automatically when T maps to a known BuiltinDType.
// If T is unknown, this will leave builtin=Unknown (still useful if you set dtype=Custom later).
template <class T>
inline void annotate(T* ptr, Metadata md = {}) {
  // If the user didn't specify dtype explicitly, keep default Builtin.
  // If they *did* specify Custom, we don't override anything here.
  md.builtin = builtin_dtype_of<std::remove_cv_t<T>>::value;
  detail::annotate_impl(static_cast<const void*>(ptr), md);
}

} // namespace mneme
