# find rocm-bundled llvm with llvm/ hints
find_package(LLVM REQUIRED CONFIG NO_DEFAULT_PATH
  HINTS "${LLVM_INSTALL_DIR}"
  PATH_SUFFIXES
    "lib/cmake/llvm"
    "lib64/cmake/llvm"
    "cmake/llvm"
    "llvm/lib/cmake/llvm"
    "llvm/lib64/cmake/llvm"
)

if(MNEME_ENABLE_HIP)
  find_package(LLD REQUIRED CONFIG NO_DEFAULT_PATH
    HINTS "${LLVM_INSTALL_PREFIX}"
  )
  message(STATUS "Found LLD package in: ${LLD_DIR}")
endif()

message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
message(STATUS "LLVM_LIBRARY_DIR: ${LLVM_LIBRARY_DIR}")
message(STATUS "LLVM_VERSION: ${LLVM_VERSION}")
message(STATUS "LLVM AVAILABLE LIBRARIES: ${LLVM_AVAILABLE_LIBS}")

if(NOT LLVM_ENABLE_RTTI)
  add_compile_options(-fno-rtti)
endif()

# use llvm-config to get library flags; we could get the lib names via llvm_map_components_to_libnames
# but would require extra parsing
set(_llvm_config "${LLVM_TOOLS_BINARY_DIR}/llvm-config")
if(NOT EXISTS "${_llvm_config}")
  message(FATAL_ERROR "llvm-config not found at ${_llvm_config}")
endif()

execute_process(
  COMMAND ${_llvm_config} --libs
  OUTPUT_VARIABLE LLVM_LIB_FLAGS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)

execute_process(
  COMMAND ${_llvm_config} --system-libs
  OUTPUT_VARIABLE LLVM_SYSTEM_LIB_FLAGS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)

if(NOT LLVM_LIB_FLAGS)
  message(FATAL_ERROR "LLVM_LIB_FLAGS is empty from llvm-config")
endif()

message(STATUS "LLVM_LIB_FLAGS: ${LLVM_LIB_FLAGS}")
message(STATUS "LLVM_SYSTEM_LIB_FLAGS: ${LLVM_SYSTEM_LIB_FLAGS}")

# Compute MNEME_LLVM_LIBS once so every target can just do
#   target_link_libraries(<tgt> PRIVATE ${MNEME_LLVM_LIBS})
# In the LLVM_LINK_LLVM_DYLIB + MNEME_LINK_SHARED_LLVM case the variable is
# empty because llvm_config(<tgt> USE_SHARED) handles linking per-target.
if(MNEME_LINK_SHARED_LLVM)
  if(NOT LLVM_LINK_LLVM_DYLIB)
    message(FATAL_ERROR
      "The LLVM installation at ${LLVM_INSTALL_PREFIX} does not provide libLLVM.so, "
      "required by MNEME_LINK_SHARED_LLVM=ON.\n"
      "Set MNEME_LINK_SHARED_LLVM=OFF.")
  endif()
  # llvm_config(<tgt> USE_SHARED) handles linking per-target; no static libs needed.
  set(MNEME_LLVM_LIBS "" CACHE INTERNAL "LLVM libs to link into Mneme targets")
else()
  if(LLVM_LINK_LLVM_DYLIB)
    message(FATAL_ERROR
      "The LLVM installation at ${LLVM_INSTALL_PREFIX} requires linking with the LLVM "
      "shared library, but MNEME_LINK_SHARED_LLVM=OFF.\n"
      "Set MNEME_LINK_SHARED_LLVM=ON.")
  endif()
  # Remove the bundled .so from the static list to avoid double-linking.
  set(_mneme_llvm_libs ${LLVM_AVAILABLE_LIBS})
  list(REMOVE_ITEM _mneme_llvm_libs "LLVM")
  set(MNEME_LLVM_LIBS ${_mneme_llvm_libs} CACHE INTERNAL "LLVM libs to link into Mneme targets")
endif()
message(STATUS "MNEME_LLVM_LIBS: ${MNEME_LLVM_LIBS}")
