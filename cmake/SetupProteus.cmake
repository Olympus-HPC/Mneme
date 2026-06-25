# proteus discovery:
# - PROTEUS_DIR:  use pre-built installation
# - PROTEUS_SRC:  use existing source tree, built in-place
# - FetchContent: clone and build version in /PROTEUS_VERSION

if(NOT PROTEUS_DIR AND NOT "$ENV{PROTEUS_DIR}" STREQUAL "")
  set(PROTEUS_DIR "$ENV{PROTEUS_DIR}" CACHE PATH "Path to a pre-built proteus installation prefix")
endif()
if(NOT PROTEUS_SRC AND NOT "$ENV{PROTEUS_SRC}" STREQUAL "")
  set(PROTEUS_SRC "$ENV{PROTEUS_SRC}" CACHE PATH "Path to a proteus source tree")
endif()

# search for existing installations (PROTEUS_DIR first)
find_package(proteus CONFIG QUIET HINTS "${PROTEUS_DIR}")

if(NOT proteus_FOUND)
  file(READ "${PROJECT_SOURCE_DIR}/PROTEUS_VERSION" _proteus_tag)
  string(STRIP "${_proteus_tag}" _proteus_tag)
  include(FetchContent)
  # forward build settings to proteus configure; LLVM_INSTALL_DIR should already be in the cache
  set(PROTEUS_ENABLE_CUDA             ${MNEME_ENABLE_CUDA})
  set(PROTEUS_ENABLE_HIP              ${MNEME_ENABLE_HIP})
  # CUDA device code cannot be compiled into a shared library; use static proteus.
  # HIP does not have this restriction, so prefer shared to avoid LLVM ODR issues.
  if(MNEME_ENABLE_CUDA)
    set(BUILD_SHARED OFF)
  else()
    set(BUILD_SHARED ON)
  endif()
  set(PROTEUS_INSTALL_IMPL_HEADERS    ON)
  set(ENABLE_TESTS                    OFF)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  if(PROTEUS_SRC)
    message(STATUS "Building proteus from PROTEUS_SRC=${PROTEUS_SRC}")
    FetchContent_Declare(proteus SOURCE_DIR "${PROTEUS_SRC}")
  else()
    message(STATUS "proteus not found -- fetching ${_proteus_tag} via FetchContent")
    FetchContent_Declare(proteus
      GIT_REPOSITORY https://github.com/Olympus-HPC/proteus.git
      GIT_TAG        ${_proteus_tag}
      GIT_SHALLOW    TRUE
    )
  endif()
  
  # in wheel mode, install the proteus cmake directory under mneme's prefix so they are bundled
  if(MNEME_PYTHON_WHEEL)
    set(CMAKE_INSTALL_LIBDIR "mneme/native/lib64")
  endif()

  FetchContent_MakeAvailable(proteus)
  # there is no install step in FetchContent, so expose the directory via proteusCore so mneme can include proteus headers
  # PROTEUS_INSTALL_IMPL_HEADERS will not help in this case
  target_include_directories(proteusCore
    INTERFACE "$<BUILD_INTERFACE:${proteus_SOURCE_DIR}/src/include>")
endif()
