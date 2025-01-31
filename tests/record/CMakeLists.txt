# The first argument is the executable to generate, the second arguments is the
# source file with checks. Any following arguments are other source files
# (without checks) to compile for generating the executable.
function(CREATE_GPU_TEST exe check_source)
  add_executable(${exe} ${check_source} ${ARGN})
  if (ENABLE_HIP)
    set_source_files_properties(${check_source} ${ARGN} PROPERTIES LANGUAGE HIP) 
    target_compile_definitions(${exe} PRIVATE ENABLE_HIP)
    target_link_libraries(${exe} PRIVATE hip::host)
  elseif(ENABLE_CUDA)
    target_compile_definitions(${exe} PRIVATE ENABLE_CUDA)
  endif()
  target_include_directories(${exe} PRIVATE ${CMAKE_SOURCE_DIR}/src/)

  set_source_files_properties(${check_source} ${ARGN} PROPERTIES
    OBJECT_DEPENDS ${PROJECT_SOURCE_DIR}/pass/RRPass.cpp
    )
  add_dependencies(${exe} RRPass)
  add_dependencies(${exe} mnemerecord)

  target_compile_options(
      ${exe}
      PUBLIC
      -fpass-plugin=$<TARGET_FILE:RRPass>
  )

  add_test(NAME ${exe} COMMAND ${LIT} -vv -D PG=1 ${check_source})
  set_tests_properties(${exe} PROPERTIES LABELS "MnemeRecord")
endfunction()


file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py "
import lit.formats

config.name = 'LIT tests'
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.cpp']
config.test_source_root = '${CMAKE_CURRENT_SOURCE_DIR}'
config.test_exec_root = '${CMAKE_CURRENT_BINARY_DIR}'
MNEME_PRELOAD_LIB='${CMAKE_BINARY_DIR}/src/libmnemerecord.so'
PAGE_SIZE = lit_config.params['PG']
config.substitutions.append(('MNEME_PRELOAD_LIB', MNEME_PRELOAD_LIB))
config.substitutions.append(('%PG', PAGE_SIZE))
"
)

CREATE_GPU_TEST(kernel kernel.cpp)
