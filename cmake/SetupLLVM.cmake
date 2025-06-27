# Add the location of LLVMConfig.cmake to CMake search paths (so that
# find_package can locate it)

list(APPEND CMAKE_PREFIX_PATH "${LLVM_INSTALL_DIR}/lib/cmake/llvm/")

# NOTE: When having just $LLVM_VERSION is not working out find_package(LLVM
# REQUIRED CONFIG HINTS "${LLVM_INSTALL_DIR}/lib/cmake/llvm/")
find_package(LLVM REQUIRED CONFIG)

message(STATUS "LLVM_FOUND: ${LLVM_FOUND}")
message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
message(STATUS "LLVM_LIBRARY_DIR: ${LLVM_LIBRARY_DIR}")
message(STATUS "LLVM_LIBRARIES: ${LLVM_LIBRARIES}")
message(STATUS "LLVM_VERSION: ${LLVM_VERSION}")
message(STATUS "LLVM_VERSION: ${LLVM_AVAILABLE_LIBS}")

if(NOT LLVM_ENABLE_RTTI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()
