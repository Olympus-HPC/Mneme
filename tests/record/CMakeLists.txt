# The first argument is the executable to generate, the second arguments is the
# source file with checks. Any following arguments are other source files
# (without checks) to compile for generating the executable.
function(CREATE_GPU_TEST exe check_source)
  add_executable(${exe} ${check_source} ${ARGN})
  if (MNEME_ENABLE_HIP)
    set_source_files_properties(${check_source} ${ARGN} PROPERTIES LANGUAGE HIP) 
    target_compile_definitions(${exe} PRIVATE MNEME_ENABLE_HIP)
    target_link_libraries(${exe} PRIVATE hip::host)
  elseif(MNEME_ENABLE_CUDA)
    target_compile_definitions(${exe} PRIVATE MNEME_ENABLE_CUDA)
  endif()
  target_include_directories(${exe} PRIVATE ${CMAKE_SOURCE_DIR}/include/)

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

add_test(NAME ${exe} COMMAND ${LIT} -a -vv -D ext='' -D PG=1 -D RR=${RECORD_READ_JSON} ${check_source})
  set_tests_properties(${exe} PROPERTIES LABELS "MnemeRecord::Link")
endfunction()

function(CREATE_RDC_GPU_TEST exe check_source)
  add_executable(${exe}.rdc ${check_source} ${ARGN})
  if (MNEME_ENABLE_HIP)
    set_source_files_properties(${check_source} ${ARGN} PROPERTIES LANGUAGE HIP) 
    target_compile_definitions(${exe}.rdc PRIVATE MNEME_ENABLE_HIP)
    target_link_libraries(${exe}.rdc PRIVATE hip::host)
  elseif(MNEME_ENABLE_CUDA)
    target_compile_definitions(${exe}.rdc PRIVATE MNEME_ENABLE_CUDA)
  endif()
  target_include_directories(${exe}.rdc PRIVATE ${CMAKE_SOURCE_DIR}/include/)

  set_source_files_properties(${check_source} ${ARGN} PROPERTIES
    OBJECT_DEPENDS ${PROJECT_SOURCE_DIR}/pass/RRPass.cpp
    )
  add_dependencies(${exe}.rdc RRPass)
  add_dependencies(${exe}.rdc mnemerecord)

  if(MNEME_ENABLE_HIP)
    # This is unsupported see: https://gitlab.kitware.com/cmake/cmake/-/issues/23210
    # set_target_properties(${exe}.rdc PROPERTIES HIP_SEPARABLE_COMPILATION on)

    # Add -fgpu-rdc, --hip-link options for compilation and linking to enable RDC.
    target_compile_options(
        ${exe}.rdc
        PUBLIC
        -fgpu-rdc
        -fpass-plugin=$<TARGET_FILE:RRPass>
    )

    target_link_options(
        ${exe}.rdc
        PUBLIC
        -fgpu-rdc
        --hip-link
        -Xoffload-linker --load-pass-plugin=$<TARGET_FILE:RRPass>
    )
  endif()

  add_test(NAME ${exe}.rdc COMMAND ${LIT} -a -vv -D ext=.rdc -D PG=1 -D RR=${RECORD_READ_JSON} ${check_source})
  set_tests_properties(${exe}.rdc PROPERTIES LABELS "MnemeRecord::RDC")
endfunction()


find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(RECORD_READ_JSON "${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/read-record.py")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/read-record.py ${CMAKE_CURRENT_BINARY_DIR}/read-record.py COPYONLY)



file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py "
import lit.formats

config.name = 'LIT tests'
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.cpp']
config.test_source_root = '${CMAKE_CURRENT_SOURCE_DIR}'
config.test_exec_root = '${CMAKE_CURRENT_BINARY_DIR}'
MNEME_PRELOAD_LIB='${CMAKE_BINARY_DIR}/src/libmnemerecord.so'
PAGE_SIZE = lit_config.params['PG']
RR = lit_config.params['RR']
EXT = lit_config.params['ext']
config.substitutions.append(('MNEME_PRELOAD_LIB', MNEME_PRELOAD_LIB))
config.substitutions.append(('%PG', PAGE_SIZE))
config.substitutions.append(('%RR', RR))
config.substitutions.append(('%ext', EXT))
"
)

CREATE_GPU_TEST(kernel kernel.cpp)
CREATE_RDC_GPU_TEST(kernel kernel.cpp)

CREATE_GPU_TEST(test_block_grid_1d test_block_grid_1d.cpp)
CREATE_RDC_GPU_TEST(test_block_grid_1d test_block_grid_1d.cpp)

CREATE_GPU_TEST(test_block_grid_2d test_block_grid_2d.cpp)
CREATE_RDC_GPU_TEST(test_block_grid_2d test_block_grid_2d.cpp)

CREATE_GPU_TEST(test_block_grid_3d test_block_grid_3d.cpp)
CREATE_RDC_GPU_TEST(test_block_grid_3d test_block_grid_3d.cpp)

CREATE_GPU_TEST(test_multi_file multi_file1.cpp multi_file2.cpp)
CREATE_RDC_GPU_TEST(test_multi_file multi_file1.cpp multi_file2.cpp)

CREATE_GPU_TEST(test_multi_file_launcher multi_file1_kernel_launcher.cpp multi_file2_kernel_launcher.cpp)
CREATE_RDC_GPU_TEST(test_multi_file_launcher multi_file1_kernel_launcher.cpp multi_file2_kernel_launcher.cpp)

CREATE_GPU_TEST(indirect_launcher_tpl_multi_arg indirect_launcher_tpl_multi_arg.cpp)
CREATE_RDC_GPU_TEST(indirect_launcher_tpl_multi_arg indirect_launcher_tpl_multi_arg.cpp)


