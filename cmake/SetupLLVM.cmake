# Copyright (c) 2021-2024. LLNL-CODE-2000766 and Mneme Contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Add the location of LLVMConfig.cmake to CMake search paths (so that
# find_package can locate it)

find_package(LLVM REQUIRED CONFIG NO_DEFAULT_PATH
  HINTS "${LLVM_INSTALL_DIR}"
  PATH_SUFFIXES
    "lib/cmake/llvm"
    "lib64/cmake/llvm"
    "cmake/llvm"
    "llvm/lib/cmake/llvm"
    "llvm/lib64/cmake/llvm"
)

file(REAL_PATH "${LLVM_DIR}" LLVM_DIR)

message(STATUS "LLVM_FOUND: ${LLVM_FOUND}")
message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
message(STATUS "LLVM_LIBRARY_DIR: ${LLVM_LIBRARY_DIR}")
message(STATUS "LLVM_LIBRARIES: ${LLVM_LIBRARIES}")
message(STATUS "LLVM_VERSION: ${LLVM_VERSION}")
message(STATUS "LLVM_VERSION: ${LLVM_AVAILABLE_LIBS}")

if(NOT LLVM_ENABLE_RTTI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()
