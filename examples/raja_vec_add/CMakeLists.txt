cmake_minimum_required(VERSION 3.8)

project(raja-example  LANGUAGES C CXX)

option(MNEME_ENABLE_CUDA "Enable CUDA" OFF)
option(MNEME_ENABLE_HIP "Enable CUDA" OFF)

find_package(RAJA REQUIRED HINTS "${RAJA_Dir}")
find_package(RR REQUIRED HINTS "${RR_Dir}")

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


add_executable(vecAdd-raja vec_add_raja.cu)

# Get the full path of the LLVM pass plugin (MyLLVMPass)
get_target_property(RRPassPath RR::RRPass LOCATION)
message(STATUS "Record Replay Pass path: ${RRPassPath}")
message(STATUS "DEVICE LANGUAGE is: ${DEVICE_LANGUAGE}")

set_source_files_properties(vec_add_raja.cu PROPERTIES LANGUAGE ${DEVICE_LANGUAGE})

target_compile_options(
 vecAdd-raja 
      PUBLIC
      $<$<COMPILE_LANGUAGE:${DEVICE_LANGUAGE}>:-fpass-plugin=${RRPassPath}>
    ) 

#set_source_files_properties(vec_add_raja.cu PROPERTIES COMPILE_FLAGS "--expt-extended-lambda")
if (MNEME_ENABLE_CUDA)
target_link_libraries(vecAdd-raja PRIVATE RAJA CUDA::cudart)
elseif(MNEME_ENABLE_HIP)
  target_include_directories(vecAdd-raja PRIVATE ${HIP_INCLUDE_DIRS})
target_link_libraries(vecAdd-raja PRIVATE RAJA hip::host)
endif()
