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
option(MNEME_ENABLE_HIP "Enable HIP" OFF)
option(MNEME_ENABLE_TESTS "Enable Testing" OFF)
option(MNEME_ENABLE_LOGGER "Enable logging on mneme" OFF)
option(MNEME_ENABLE_PYTHON "Build Python bindings, profiling, and autotuning" OFF)
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
  find_library(MNEME_NVPTXCOMPILER_LIBRARY
    NAMES nvptxcompiler_static libnvptxcompiler_static nvptxcompiler
    REQUIRED
    NO_DEFAULT_PATH
    PATHS "${CUDAToolkit_LIBRARY_DIR}"
  )
  message(STATUS "CUDA nvPTXCompiler library: ${MNEME_NVPTXCOMPILER_LIBRARY}")
  add_definitions("-DMNEME_ENABLE_CUDA")
elseif(MNEME_ENABLE_HIP)
  find_package(HIP REQUIRED)
  enable_language(HIP)
  add_definitions("-DMNEME_ENABLE_HIP")
else()
  message(FATAL_ERROR "Mneme requires either CUDA or HIP to be enabled")
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
if(MNEME_ENABLE_PYTHON)
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

# Python bindings: cmake builds c++ libs, pip installs python package
if(MNEME_ENABLE_PYTHON)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  message(STATUS "Python package installation enabled")

  # pip helper options (can be overridden by spack or other users)
  set(PY_PIP_OPTIONS "-v" CACHE STRING
      "Additional options for pip wheel/install commands")
  set(PY_PIP_INSTALL_OPTIONS "" CACHE STRING
      "Additional options for pip install (prefix, etc.)")

  # Configure config.json from template for pip to include in the wheel
  # Collect LLVM library information
  string(REPLACE ";" " " LLVM_LIBS "${LLVM_LIBRARIES}")
  if(LLVM_SYSTEM_LIBS)
    string(REPLACE ";" " " LLVM_SYSTEM_LIBS "${LLVM_SYSTEM_LIBS}")
  else()
    set(LLVM_SYSTEM_LIBS "")
  endif()

  # Set PREFIX as a literal string so we can process it in Python later
  set(PREFIX "@PREFIX@")

  # Generate config.json in the source tree so pip can include it in the wheel
  configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/config.json.in"
    "${PROJECT_SOURCE_DIR}/python/mneme/native/config.json"
    @ONLY
  )

  # Build a wheel by pointing pip to our pre-built C++ libraries
  add_custom_target(pip_wheel
    ${CMAKE_COMMAND} -E rm -rf mneme-whl pip-staging
    # do a local install to a staging dir (with libs, headers, cmake)
    COMMAND
      ${CMAKE_COMMAND} --install . --prefix ${CMAKE_BINARY_DIR}/pip-staging
    COMMAND
      ${CMAKE_COMMAND} -E env
        MNEME_LIB_DIR=${CMAKE_BINARY_DIR}/pip-staging
        LLVM_INSTALL_DIR=${LLVM_INSTALL_DIR}
      ${Python3_EXECUTABLE} -m pip ${PY_PIP_OPTIONS} wheel
        --no-build-isolation --no-deps --wheel-dir=mneme-whl
        "${PROJECT_SOURCE_DIR}"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS mneme mneme_profile record mnemert
    COMMENT "Building Python wheel with pre-built C++ libraries"
  )

  # Install the wheel with dependencies (normal manual build)
  add_custom_target(pip_install
    ${CMAKE_COMMAND} -E env
      LLVM_INSTALL_DIR=${LLVM_INSTALL_DIR}
    ${Python3_EXECUTABLE} -m pip ${PY_PIP_OPTIONS} install
      --force-reinstall --no-index --no-deps ${PY_PIP_INSTALL_OPTIONS}
      --find-links=mneme-whl mneme
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS pip_wheel
    COMMENT "Installing Python package with dependencies"
  )

  # Install the wheel without dependencies (spack-like builds)
  add_custom_target(pip_install_nodeps
    ${CMAKE_COMMAND} -E env
      LLVM_INSTALL_DIR=${LLVM_INSTALL_DIR}
    ${Python3_EXECUTABLE} -m pip ${PY_PIP_OPTIONS} install
      --force-reinstall --no-index --no-deps ${PY_PIP_INSTALL_OPTIONS}
      --find-links=mneme-whl mneme
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS pip_wheel
    COMMENT "Installing Python package without dependencies"
  )
endif()
