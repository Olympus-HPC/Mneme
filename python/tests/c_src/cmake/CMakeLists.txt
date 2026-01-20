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
add_executable(vecAdd vec_add.cu)

add_mneme(vecAdd)

if (RDC)
if(ENABLE_CUDA)
  target_link_options(vecAdd  PUBLIC -fgpu-rdc)
elseif(ENABLE_HIP)
  target_link_options(vecAdd  PUBLIC -fgpu-rdc --hip-link)
endif()
endif()

if(ENABLE_CUDA)
set_source_files_properties(vec_add.cu PROPERTIES LANGUAGE CUDA)
target_link_libraries(vecAdd PRIVATE CUDA::cudart)
elseif(ENABLE_HIP)
set_source_files_properties(vec_add.cu PROPERTIES LANGUAGE HIP)
target_include_directories(vecAdd PRIVATE ${HIP_INCLUDE_DIRS})
target_link_libraries(vecAdd PRIVATE hip::host)
endif()

