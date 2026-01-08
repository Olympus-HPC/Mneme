# Where the generated header will live
set(BUILD_INFO_HPP "${CMAKE_CURRENT_BINARY_DIR}/src/miniFE_info.hpp")
set(miniFE_src ${CMAKE_CURRENT_SOURCE_DIR}/basic/BoxPartition.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/YAML_Doc.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/YAML_Element.cpp ${CMAKE_CURRENT_SOURCE_DIR}/utils/param_utils.cpp ${CMAKE_CURRENT_SOURCE_DIR}/utils/utils.cpp ${CMAKE_CURRENT_SOURCE_DIR}/utils/mytimer.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
add_executable(miniFE ${miniFE_src})
add_mneme(miniFE)
target_include_directories(miniFE PRIVATE ${PROJECT_SOURCE_DIR}/include)

target_compile_definitions(miniFE PRIVATE  MINIFE_CSR_MATRIX MINIFE_SCALAR=double MINIFE_LOCAL_ORDINAL=int MINIFE_GLOBAL_ORDINAL=int MINIFE_RESTRICT=__restrict__ MINIFE_INFO=0 MINIFE_KERNELS=0)
target_include_directories(miniFE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/ ${CMAKE_CURRENT_SOURCE_DIR}/utils/ ${CMAKE_CURRENT_SOURCE_DIR}/fem/ ${CMAKE_CURRENT_SOURCE_DIR}/basic/)
if (WITH_MNEME_EXAMPLE_HIP)
  set_source_files_properties(${miniFE_src} PROPERTIES LANGUAGE HIP)
  target_link_libraries(miniFE PRIVATE hip::device hip::host)
  target_include_directories(miniFE PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_compile_definitions(miniFE PRIVATE MNEME_EXAMPLE_ENABLE_HIP)
elseif(WITH_MNEME_EXAMPLE_CUDA)
endif()

add_test(
  NAME MINIFE 
  COMMAND miniFE -nx 220 -ny 200 -nz 190 
)
