add_executable(bezier main.cpp)
add_mneme(bezier)
target_compile_definitions(bezier PRIVATE MKX=4)
if (WITH_MNEME_EXAMPLE_HIP)
  set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
  target_link_libraries(bezier PRIVATE hip::device hip::host)
  target_include_directories(bezier PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_compile_definitions(bezier PRIVATE MNEME_EXAMPLE_ENABLE_HIP)
elseif(WITH_MNEME_EXAMPLE_CUDA)
endif()

add_test(
  NAME BEZIER 
  COMMAND bezier -f ${CMAKE_CURRENT_SOURCE_DIR}/input/control.txt  -n 8192 
)
