#pragma once

#include "mneme/MnemeAnnotation.hpp"

extern "C" {
bool mneme_set_metadata_for_ptr(const void *ptr, mneme::Metadata md)
    __attribute__((weak));
bool mneme_get_metadata_for_ptr(const void *ptr, mneme::Metadata *md)
    __attribute__((weak));
bool mneme_erase_metadata_for_ptr(const void *ptr) __attribute__((weak));
}

namespace mneme {
namespace detail {

void annotate_impl(const void *ptr, Metadata md);
bool get_annotation(const void *ptr, Metadata &md);
void erase_annotation(const void *ptr);

} // namespace detail
} // namespace mneme