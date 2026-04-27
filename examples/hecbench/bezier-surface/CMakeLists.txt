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
  COMMAND ${target-name} -f ${CMAKE_CURRENT_SOURCE_DIR}/input/control.txt  -n 8192 
)
endfunction()


add_executable(bezier-mneme main.cpp)
add_mneme(bezier-mneme)
BUILD(bezier-mneme BEZIER::MNEME)

add_executable(bezier-orig main.cpp)
BUILD(bezier-orig BEZIER::ORIG)

add_executable(bezier-proteus main.cpp)
target_compile_definitions(bezier-proteus PRIVATE ENABLE_PROTEUS)
add_proteus(bezier-proteus)
BUILD(bezier-proteus BEZIER::PROTEUS)
