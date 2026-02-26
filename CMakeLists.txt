cmake_minimum_required(VERSION 3.18)

project(
  mneme
  VERSION 0.1.0
  LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD
    17
    CACHE STRING "")
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(MNEME_ENABLE_CUDA "Enable CUDA" OFF)
option(MNEME_ENABLE_HIP "Enable CUDA" OFF)
option(MNEME_ENABLE_TESTS "Enable Testing" OFF)
option(MNEME_ENABLE_LOGGER "Enable logging on mneme" OFF)
option(MNEME_ENABLE_AUTOTUNE "Enable autotuning and generate python" OFF)
option(MNEME_LINK_SHARED_LLVM "Link with the LLVM shared library" OFF)

if(MNEME_ENABLE_LOGGER)
  find_package(spdlog REQUIRED)
endif()

message(STATUS "Installation prefix is ${CMAKE_INSTALL_PREFIX}")

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

if(MNEME_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  find_package(CUDA REQUIRED)
  enable_language(CUDA)
  message(STATUS "Cuda Libraries: ${CUDA_LIBRARIES}")
  message(STATUS "Cuda Library: ${CUDA_LIBRARY}")
  add_definitions("-DMNEME_ENABLE_CUDA")
elseif(MNEME_ENABLE_HIP)
  find_package(HIP REQUIRED)
  enable_language(HIP)
  add_definitions("-DMNEME_ENABLE_HIP")
else()
  message(FATAL "Mneme supports either CUDA OR HIP")
endif()

include(cmake/SetupLLVM.cmake)
find_package(proteus REQUIRED)
include(cmake/MnemeFunctions.cmake)
get_target_property(proteus_link_dirs proteus INTERFACE_LINK_DIRECTORIES)


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
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/mneme")

write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/mnemeConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion)

set(MNEME_TARGETS mnemert record)
if(MNEME_ENABLE_AUTOTUNE)
  list(APPEND MNEME_TARGETS mneme_profile mneme)
endif()

install(
  TARGETS ${MNEME_TARGETS} 
  EXPORT mnemeTargets
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

install(
  FILES "${PROJECT_SOURCE_DIR}/include/mneme/MnemeAnnotation.hpp"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/mneme")

# Install and export CMake package
install(
  EXPORT mnemeTargets
  FILE mnemeTargets.cmake
  NAMESPACE Mneme::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mneme)

install(FILES "${PROJECT_BINARY_DIR}/mnemeConfigVersion.cmake"
              "${PROJECT_BINARY_DIR}/mnemeConfig.cmake"
        "${PROJECT_SOURCE_DIR}/cmake/MnemeFunctions.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mneme)
