add_library(RRPass SHARED RRPass.cpp)
add_library(RR::Pass ALIAS RRPass)

target_include_directories(RRPass
  SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})

if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(RRPass PRIVATE -fno-rtti)
endif()

llvm_map_components_to_libnames(llvm_libs all)

if (MNEME_ENABLE_CUDA)
target_compile_definitions(RRPass PRIVATE -DMNEME_ENABLE_CUDA -DMNEME_ENABLE_DEBUG)
elseif(MNEME_ENABLE_HIP)
  target_compile_definitions(RRPass PRIVATE -DMNEME_ENABLE_HIP -DMNEME_ENABLE_DEBUG)
endif()

target_link_libraries(RRPass PRIVATE ${llvm_libs})
target_include_directories(RRPass PRIVATE "${PROJECT_SOURCE_DIR}/src")

install(
  TARGETS RRPass
  EXPORT mnemeTargets
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  )

