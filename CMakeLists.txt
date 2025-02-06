cmake_minimum_required(VERSION 3.18)

project(mneme
  VERSION 0.1.0
  LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "")
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


string(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_UPPER)
if(BUILD_TYPE_UPPER STREQUAL "DEBUG")
  set(CMAKE_CXX_FLAGS_DEBUG_INIT "-gdwarf-4" CACHE STRING "Flags used by the compiler during debug builds" FORCE)
  set(CMAKE_C_FLAGS_DEBUG_INIT "-gdwarf-4" CACHE STRING "Flags used by the compiler during debug builds" FORCE)
  set(CMAKE_CUDA_FLAGS_DEBUG_INIT "-gdwarf-4" CACHE STRING "Flags used by the compiler during debug builds" FORCE)
  message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG_INIT}")
  message(STATUS "CMAKE_C_FLAGS_DEBUG: ${CMAKE_C_FLAGS_DEBUG_INIT}")
  message(STATUS "CMAKE_CUDA_FLAGS_DEBUG: ${CMAKE_CUDA_FLAGS_DEBUG_INIT}")
  add_compile_options("-gdwarf-4")
elseif(BUILD_TYPE_UPPER STREQUAL "RELWITHDEBINFO")
  set(CMAKE_CXX_FLAGS_DEBUG_INIT "-gdwarf-4" CACHE STRING "Flags used by the compiler during debug builds" FORCE)
  set(CMAKE_C_FLAGS_DEBUG_INIT "-gdwarf-4" CACHE STRING "Flags used by the compiler during debug builds" FORCE)
  set(CMAKE_CUDA_FLAGS_DEBUG_INIT "-gdwarf-4" CACHE STRING "Flags used by the compiler during debug builds" FORCE)
  message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG_INIT}")
  message(STATUS "CMAKE_C_FLAGS_DEBUG: ${CMAKE_C_FLAGS_DEBUG_INIT}")
  message(STATUS "CMAKE_CUDA_FLAGS_DEBUG: ${CMAKE_CUDA_FLAGS_DEBUG_INIT}")
  add_compile_options("-gdwarf-4")
endif()

option(MNEME_ENABLE_CUDA "Enable CUDA" OFF)
option(MNEME_ENABLE_HIP "Enable CUDA" OFF)
option(MNEME_ENABLE_TESTS "Enable Testing" OFF)
option(MNEME_ENABLE_DEBUG "Enable Debug Output" OFF)
option(MNEME_ENABLE_LOGGER "Enable logging on mneme" ON)

find_package(proteus REQUIRED)
get_target_property(proteus_link_dirs proteus INTERFACE_LINK_DIRECTORIES)
message(STATUS "Proteus Link Directories: ${proteus_link_dirs}")

if (MNEME_ENABLE_LOGGER)
find_package(spdlog REQUIRED)
endif()

message(STATUS "Installation prefix is ${CMAKE_INSTALL_PREFIX}")

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

if(MNEME_ENABLE_CUDA)
set(LLVM_VERSION "18.1.8" CACHE STRING "LLVM version")
  enable_language(CUDA)
  add_definitions("-DMNEME_ENABLE_CUDA")
  find_package(CUDAToolkit REQUIRED)
  find_package(CUDA REQUIRED)
  set(CMAKE_CUDA_COMPILER "clang++")
  message(STATUS "Cuda Libraries: ${CUDA_LIBRARIES}")
  message(STATUS "Cuda Library: ${CUDA_LIBRARY}")
elseif(MNEME_ENABLE_HIP)
set(LLVM_VERSION "18" CACHE STRING "LLVM version")
  enable_language(HIP)
  find_package(HIP REQUIRED)
  add_definitions("-DMNEME_ENABLE_HIP")
endif()

if(MNEME_ENABLE_DEBUG)
  add_definitions("-DMNEME_ENABLE_DEBUG")
endif()

include(cmake/SetupLLVM.cmake)

add_subdirectory(pass)
add_subdirectory(src)

if(MNEME_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# INSTALL
message(STATUS "Installing ${PROJECT_NAME} of version ${PROJECT_VERSION}")

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/mnemeConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/mnemeConfig.cmake"
  INSTALL_DESTINATION
  "${CMAKE_INSTALL_LIBDIR}/cmake/mneme"
)

write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/mnemeConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion)


# Install and export CMake package
install(EXPORT mnemeTargets
    FILE mnemeTargets.cmake
    NAMESPACE Mneme::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mneme
)

install(FILES
  "${PROJECT_BINARY_DIR}/mnemeConfigVersion.cmake"
  "${PROJECT_BINARY_DIR}/mnemeConfig.cmake"
  DESTINATION
  ${CMAKE_INSTALL_LIBDIR}/cmake/mneme)
