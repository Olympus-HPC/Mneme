cmake_minimum_required(VERSION 3.8)

project(hip-example  LANGUAGES C CXX)
option(RDC "Enable RDC compilation" OFF)
option(ENABLE_CUDA "Enable CUDA" OFF)
option(ENABLE_HIP "Enable HIP" OFF)


set(CMAKE_BUILD_TYPE RelWithDebInfo)

if(ENABLE_CUDA)
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

get_cmake_property(_targets TARGETS)
message(STATUS "Available targets: ${_targets}")

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
target_link_libraries(vecAdd PRIVATE CUDA::cudart CUDA::cuda_driver)
elseif(ENABLE_HIP)
set_source_files_properties(vec_add.cu PROPERTIES LANGUAGE HIP)
target_include_directories(vecAdd PRIVATE ${HIP_INCLUDE_DIRS})
target_link_libraries(vecAdd PRIVATE hip::host)
endif()

