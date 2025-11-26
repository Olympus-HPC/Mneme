cmake_minimum_required(VERSION 3.8)

project(hip-example  LANGUAGES C CXX)
option(RDC "Enable RDC compilation" OFF)

enable_language(HIP)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
find_package(HIP REQUIRED)
find_package(mneme REQUIRED)
add_definitions("-DMNEME_ENABLE_HIP")
add_executable(vecAdd vec_add.cu)

get_cmake_property(_targets TARGETS)
message(STATUS "Available targets: ${_targets}")

add_mneme(vecAdd)
if (RDC)
        target_link_options(vecAdd  PUBLIC -fgpu-rdc --hip-link)
endif()

set_source_files_properties(vec_add.cu PROPERTIES LANGUAGE HIP)
target_include_directories(vecAdd PRIVATE ${HIP_INCLUDE_DIRS})
target_link_libraries(vecAdd PRIVATE hip::host)

