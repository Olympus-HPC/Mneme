add_library(regdeviceir SHARED regdeviceir.cpp)

target_include_directories(regdeviceir
  SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)

if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(regdeviceir PRIVATE -fno-rtti)
endif()

llvm_map_components_to_libnames(llvm_libs all)

if (MNEME_ENABLE_CUDA)
target_compile_definitions(regdeviceir PRIVATE -DMNEME_ENABLE_CUDA -DMNEME_ENABLE_DEBUG)
elseif(MNEME_ENABLE_HIP)
  target_compile_definitions(regdeviceir PRIVATE -DMNEME_ENABLE_HIP -DMNEME_ENABLE_DEBUG)
endif()

target_link_libraries(regdeviceir PRIVATE ${llvm_libs})
target_include_directories(regdeviceir PRIVATE "${PROJECT_SOURCE_DIR}/src")

target_link_libraries(regdeviceir
  "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")

target_compile_options(
 regdeviceir 
  INTERFACE
  -fpass-plugin=$<TARGET_FILE:Mneme::regdeviceir>
)

target_link_options(
  regdeviceir 
  INTERFACE
  "SHELL:$<$<LINK_LANGUAGE:HIP>:-Xoffload-linker --load-pass-plugin=$<TARGET_FILE:Mneme::regdeviceir>>"
)

