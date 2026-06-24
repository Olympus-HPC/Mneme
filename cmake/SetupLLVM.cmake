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
message(STATUS "LLVM_LINK_LLVM_DYLIB: ${LLVM_LINK_LLVM_DYLIB}")

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

# Validate that the requested linking mode is compatible with the LLVM installation,
# then compute MNEME_LLVM_LIBS so every target can do:
#   target_link_libraries(<tgt> PRIVATE ${MNEME_LLVM_LIBS})
# When MNEME_LINK_SHARED_LLVM=ON the variable is empty because
# llvm_config(<tgt> USE_SHARED) handles per-target linking.
if(MNEME_LINK_SHARED_LLVM)
  if(NOT LLVM_LINK_LLVM_DYLIB)
    message(FATAL_ERROR
      "MNEME_LINK_SHARED_LLVM=ON but the LLVM installation at ${LLVM_LIBRARY_DIR} "
      "was not built with shared library support "
      "(LLVM_LINK_LLVM_DYLIB is not set in LLVMConfig.cmake).")
  endif()
  find_library(_mneme_llvm_shared_lib
    NAMES LLVM
    PATHS "${LLVM_LIBRARY_DIR}"
    NO_DEFAULT_PATH
  )
  if(NOT _mneme_llvm_shared_lib)
    message(FATAL_ERROR
      "LLVM_LINK_LLVM_DYLIB=ON but libLLVM.so was not found in ${LLVM_LIBRARY_DIR}.")
  endif()
  message(STATUS "LLVM linking mode: SHARED (${_mneme_llvm_shared_lib})")
  # llvm_config(<tgt> USE_SHARED) handles linking per-target; no static libs needed.
  set(MNEME_LLVM_LIBS "" CACHE INTERNAL "LLVM libs to link into Mneme targets")
else()
  if(LLVM_LINK_LLVM_DYLIB)
    message(FATAL_ERROR
      "MNEME_LINK_SHARED_LLVM=OFF but the LLVM installation at ${LLVM_LIBRARY_DIR} "
      "was built for shared linking (LLVM_LINK_LLVM_DYLIB=ON in LLVMConfig.cmake)")
  endif()
  message(STATUS "LLVM linking mode: STATIC")
  # Remove the bundled .so target from the static list to avoid double-linking.
  set(_mneme_llvm_libs ${LLVM_AVAILABLE_LIBS})
  list(REMOVE_ITEM _mneme_llvm_libs "LLVM")
  set(MNEME_LLVM_LIBS ${_mneme_llvm_libs} CACHE INTERNAL "LLVM libs to link into Mneme targets")
endif()
message(STATUS "MNEME_LLVM_LIBS: ${MNEME_LLVM_LIBS}")
