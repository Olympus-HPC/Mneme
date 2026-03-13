#include "MnemeAnnotationRuntime.hpp"

namespace mneme {

void annotate(const void *ptr, Metadata md) {
  detail::annotate_impl(ptr, std::move(md));
}

void annotate(void *ptr, Metadata md) {
  detail::annotate_impl(ptr, std::move(md));
}

namespace detail {

void annotate_impl(const void *ptr, Metadata md) {
  if (!ptr)
    return;

  if (mneme_set_metadata_for_ptr)
    mneme_set_metadata_for_ptr(ptr, std::move(md));
}

bool get_annotation(const void *ptr, Metadata &md) {
  if (!ptr)
    return false;

  if (!mneme_get_metadata_for_ptr)
    return false;

  return mneme_get_metadata_for_ptr(ptr, &md);
}

void erase_annotation(const void *ptr) {
  if (!ptr)
    return;

  if (mneme_erase_metadata_for_ptr)
    mneme_erase_metadata_for_ptr(ptr);
}

} // namespace detail
} // namespace mneme
