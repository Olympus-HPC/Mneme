cmake_minimum_required(VERSION 3.8)

project(cuda-example  LANGUAGES C CXX)

option(MNEME_ENABLE_CUDA "Enable CUDA" OFF)
option(MNEME_ENABLE_HIP "Enable CUDA" OFF)


if(MNEME_ENABLE_CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED)
add_definitions("-DMNEME_ENABLE_CUDA")
set(DEVICE_LANGUAGE "CUDA")
elseif(MNEME_ENABLE_HIP)
enable_language(HIP)
find_package(HIP REQUIRED)
add_definitions("-DMNEME_ENABLE_HIP")
set(DEVICE_LANGUAGE "HIP")
else()
  message(FATAL_ERROR "Please define either '-DMNEME_ENABLE_HIP' or -DMNEME_ENABLE_CUDA")
endif()


add_executable(vecAdd vec_add.cu)

# Get the full path of the LLVM pass plugin (MyLLVMPass)

set_source_files_properties(vec_add.cu PROPERTIES LANGUAGE ${DEVICE_LANGUAGE})
set(RRPassPath ${RR_Dir}/pass/libRRPass.so)
message(STATUS "Pass is ${RRPassPath}")

if (MNEME_ENABLE_CUDA)
target_link_libraries(vecAdd PRIVATE CUDA::cudart)
elseif(MNEME_ENABLE_HIP)
  target_include_directories(vecAdd PRIVATE ${HIP_INCLUDE_DIRS})
target_link_libraries(vecAdd PRIVATE hip::host)
endif()

target_compile_options(
 vecAdd
      PUBLIC
      $<$<COMPILE_LANGUAGE:HIP>:-fpass-plugin=${RRPassPath}>
    ) 

