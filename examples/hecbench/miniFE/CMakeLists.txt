
function(BUILD target-name test-name)
target_include_directories(${target-name} PRIVATE ${PROJECT_SOURCE_DIR}/include)

target_compile_definitions(${target-name} PRIVATE  MINIFE_CSR_MATRIX MINIFE_SCALAR=double MINIFE_LOCAL_ORDINAL=int MINIFE_GLOBAL_ORDINAL=int MINIFE_RESTRICT=__restrict__ MINIFE_INFO=0 MINIFE_KERNELS=0)
target_include_directories(${target-name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/ ${CMAKE_CURRENT_SOURCE_DIR}/utils/ ${CMAKE_CURRENT_SOURCE_DIR}/fem/ ${CMAKE_CURRENT_SOURCE_DIR}/basic/)
if (WITH_MNEME_EXAMPLE_HIP)
  set_source_files_properties(${miniFE_src} PROPERTIES LANGUAGE HIP)
  target_link_libraries(${target-name} PRIVATE hip::device hip::host)
  target_include_directories(${target-name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_compile_definitions(${target-name} PRIVATE MNEME_EXAMPLE_ENABLE_HIP)
elseif(WITH_MNEME_EXAMPLE_CUDA)
    set_source_files_properties(${miniFE_src} PROPERTIES LANGUAGE CUDA)
    target_link_libraries(${target-name} PRIVATE CUDA::cudart CUDA::cuda_driver)
    target_include_directories(${target-name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_compile_definitions(${target-name} PRIVATE MNEME_EXAMPLE_ENABLE_CUDA)
    set_target_properties(${target-name} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
endif()

add_test(
  NAME ${test-name} 
  COMMAND ${target-name} -nx 220 -ny 200 -nz 190 
)

endfunction()


set(miniFE_src ${CMAKE_CURRENT_SOURCE_DIR}/basic/BoxPartition.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/YAML_Doc.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/YAML_Element.cpp ${CMAKE_CURRENT_SOURCE_DIR}/utils/param_utils.cpp ${CMAKE_CURRENT_SOURCE_DIR}/utils/utils.cpp ${CMAKE_CURRENT_SOURCE_DIR}/utils/mytimer.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

add_executable(miniFE-orig ${miniFE_src})
BUILD(miniFE-orig MINIFE::ORIG)

add_executable(miniFE-mneme ${miniFE_src})
add_mneme(miniFE-mneme)
BUILD(miniFE-mneme MINIFE::MNEME)

add_executable(miniFE-proteus ${miniFE_src})
target_compile_definitions(miniFE-proteus PRIVATE ENABLE_PROTEUS)
add_proteus(miniFE-proteus)
BUILD(miniFE-proteus MINIFE::PROTEUS)
