add_executable(wsm5 main.cpp)
add_mneme(wsm5)
target_compile_definitions(wsm5 PRIVATE MKX=4)
if (WITH_MNEME_EXAMPLE_HIP)
  set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
  target_link_libraries(wsm5 PRIVATE hip::device hip::host)
  target_include_directories(wsm5 PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_compile_definitions(wsm5 PRIVATE MNEME_EXAMPLE_ENABLE_HIP)
elseif(WITH_MNEME_EXAMPLE_CUDA)
endif()

add_test(
  NAME WSM5 
  COMMAND wsm5 1 
)
