
function(BUILD target-name test-name)
  target_compile_definitions(${target-name} PRIVATE MKX=4)
  if (WITH_MNEME_EXAMPLE_HIP)
    set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
    target_link_libraries(${target-name} PRIVATE hip::device hip::host)
    target_include_directories(${target-name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_compile_definitions(${target-name} PRIVATE MNEME_EXAMPLE_ENABLE_HIP)
  elseif(WITH_MNEME_EXAMPLE_CUDA)
    set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
    target_link_libraries(${target-name} PRIVATE CUDA::cudart CUDA::cuda_driver)
    target_include_directories(${target-name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_compile_definitions(${target-name} PRIVATE MNEME_EXAMPLE_ENABLE_CUDA)
    set_target_properties(${target-name} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
  endif()

  add_test(
  NAME ${test-name} 
  COMMAND ${target-name} 1 
  )
endfunction()


add_executable(wsm5-orig main.cpp)
BUILD(wsm5-orig WSM5::ORIG)

add_executable(wsm5-mneme main.cpp)
add_mneme(wsm5-mneme)
BUILD(wsm5-mneme WSM5::MNEME)

add_executable(wsm5-proteus main.cpp)
target_compile_definitions(wsm5-proteus PRIVATE ENABLE_PROTEUS)
add_proteus(wsm5-proteus LINK_SHARED)
BUILD(wsm5-proteus WSM5::PROTEUS)
