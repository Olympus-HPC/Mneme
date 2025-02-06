cmake_minimum_required(VERSION 3.8)

project(hip-example  LANGUAGES C CXX)

enable_language(HIP)
find_package(HIP REQUIRED)
find_package(mneme REQUIRED)
add_definitions("-DMNEME_ENABLE_HIP")
add_executable(vecAdd vec_add.cu)

set_source_files_properties(vec_add.cu PROPERTIES LANGUAGE HIP)
message(STATUS "Pass is ${RRPassPath}")

target_include_directories(vecAdd PRIVATE ${HIP_INCLUDE_DIRS})
target_link_libraries(vecAdd PRIVATE hip::host)

