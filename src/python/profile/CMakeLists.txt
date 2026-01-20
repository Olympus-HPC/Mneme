if (MNEME_ENABLE_HIP)
add_library(mneme_profile SHARED rocr_profile.cpp)

if (MNEME_ENABLE_LOGGER)
target_link_libraries(mneme_profile PRIVATE spdlog::spdlog_header_only)
target_compile_definitions(mneme_profile PRIVATE MNEME_ENABLE_LOGGER)
endif()

find_package(rocprofiler-sdk QUIET CONFIG)
target_include_directories(mneme_profile PRIVATE ${HIP_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mneme_profile PRIVATE hip::host)


if(rocprofiler-sdk_FOUND AND TARGET rocprofiler-sdk::rocprofiler-sdk)
  target_link_libraries(mneme_profile  PRIVATE rocprofiler-sdk::rocprofiler-sdk dl pthread)
else()
  message(WARNING "CMake package for rocprofiler-sdk not found; falling back to manual include/lib lookup")
  find_path(ROCPROFILER_SDK_INCLUDE_DIR rocprofiler-sdk/rocprofiler.h
            HINTS "${ROCM_PATH}/include")
  find_library(ROCPROFILER_SDK_LIBRARY NAMES rocprofiler-sdk
               HINTS "${ROCM_PATH}/lib" "${ROCM_PATH}/lib64")
  if(NOT ROCPROFILER_SDK_INCLUDE_DIR OR NOT ROCPROFILER_SDK_LIBRARY)
    message(FATAL_ERROR "Could not find rocprofiler-sdk headers or library")
  endif()
  target_include_directories(mneme_profile PRIVATE "${ROCPROFILER_SDK_INCLUDE_DIR}")
  target_link_libraries(mneme_profile PRIVATE "${ROCPROFILER_SDK_LIBRARY}" dl pthread)
endif()
elseif(MNEME_ENABLE_CUDA)
add_library(mneme_profile SHARED cupti_profile.cpp)
endif()

add_library(mneme::profile ALIAS mneme_profile)
