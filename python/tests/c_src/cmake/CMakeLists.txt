cmake_minimum_required(VERSION 3.8)

project(hip-example  LANGUAGES C CXX)
option(RDC "Enable RDC compilation" OFF)
option(ENABLE_CUDA "Enable CUDA" OFF)
option(ENABLE_HIP "Enable HIP" OFF)

set(CMAKE_CXX_STANDARD
    17
    CACHE STRING "")
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE RelWithDebInfo)

if(ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  find_package(CUDA REQUIRED)
  add_definitions("-D__ENABLE_CUDA__")
elseif(ENABLE_HIP)
  enable_language(HIP)
  find_package(HIP REQUIRED)
  add_definitions("-D__ENABLE_HIP__")
endif()

find_package(mneme REQUIRED)

set(TEST_TARGET_NAME vecAdd CACHE STRING "Name of the test executable")
set(TEST_SOURCE_FILE vec_add.cu CACHE STRING "CUDA/HIP source file for the test executable")

add_executable(${TEST_TARGET_NAME} ${TEST_SOURCE_FILE})

add_mneme(${TEST_TARGET_NAME})

if (RDC)
if(ENABLE_CUDA)
  target_link_options(${TEST_TARGET_NAME}  PUBLIC -fgpu-rdc)
elseif(ENABLE_HIP)
  target_link_options(${TEST_TARGET_NAME}  PUBLIC -fgpu-rdc --hip-link)
endif()
endif()

if(ENABLE_CUDA)
set_source_files_properties(${TEST_SOURCE_FILE} PROPERTIES LANGUAGE CUDA)
target_link_libraries(${TEST_TARGET_NAME} PRIVATE CUDA::cudart CUDA::cuda_driver)
elseif(ENABLE_HIP)
set_source_files_properties(${TEST_SOURCE_FILE} PROPERTIES LANGUAGE HIP)
target_include_directories(${TEST_TARGET_NAME} PRIVATE ${HIP_INCLUDE_DIRS})
target_link_libraries(${TEST_TARGET_NAME} PRIVATE hip::host)
endif()

