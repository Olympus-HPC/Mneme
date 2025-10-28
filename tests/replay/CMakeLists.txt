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
    set_source_files_properties(${check_source} ${ARGN} PROPERTIES LANGUAGE CUDA) 
    target_compile_definitions(${exe} PRIVATE MNEME_ENABLE_CUDA)
    target_link_libraries(${exe} PRIVATE CUDA::cudart CUDA::cuda_driver)
  endif()
  target_include_directories(${exe} PRIVATE ${CMAKE_SOURCE_DIR}/include/)

  add_dependencies(${exe} record)
  add_dependencies(${exe} replay)
  add_mneme(${exe})

add_test(NAME ${exe} COMMAND ${LIT} -a -vv -DFILECHECK=${FILECHECK} -D ext='' -D PG=1 -D RR=${RECORD_READ_JSON} ${check_source})
set_tests_properties(${exe} PROPERTIES LABELS "MnemeReplay::Link")
endfunction()

function(CREATE_RDC_GPU_TEST exe check_source)
  add_executable(${exe}.rdc ${check_source} ${ARGN})
  if (MNEME_ENABLE_HIP)
    set_source_files_properties(${check_source} ${ARGN} PROPERTIES LANGUAGE HIP) 
    target_compile_definitions(${exe}.rdc PRIVATE MNEME_ENABLE_HIP)
    target_link_libraries(${exe}.rdc PRIVATE hip::host)
  elseif(MNEME_ENABLE_CUDA)
    set_source_files_properties(${check_source} ${ARGN} PROPERTIES LANGUAGE CUDA) 
    target_compile_definitions(${exe}.rdc PRIVATE MNEME_ENABLE_CUDA)
    target_link_libraries(${exe}.rdc PRIVATE CUDA::cudart CUDA::cuda_driver)
  endif()
  target_include_directories(${exe}.rdc PRIVATE ${CMAKE_SOURCE_DIR}/include/)

  add_dependencies(${exe}.rdc record)
  add_dependencies(${exe}.rdc replay)
  add_mneme(${exe}.rdc)

  if(MNEME_ENABLE_HIP)
    # This is unsupported see: https://gitlab.kitware.com/cmake/cmake/-/issues/23210
    # set_target_properties(${exe}.rdc PROPERTIES HIP_SEPARABLE_COMPILATION on)

    # Add -fgpu-rdc, --hip-link options for compilation and linking to enable RDC.
    target_compile_options(
        ${exe}.rdc
        PUBLIC
        -fgpu-rdc
    )

    target_link_options(
        ${exe}.rdc
        PUBLIC
        -fgpu-rdc
        --hip-link
    )
    elseif(MNEME_ENABLE_CUDA)
        target_compile_options(
            ${exe}.rdc
            PUBLIC
        )
    set_target_properties(${exe}.rdc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  endif()

  add_test(NAME ${exe}.rdc COMMAND ${LIT} -a -vv -DFILECHECK=${FILECHECK} -D ext=.rdc -D PG=1 -D RR=${RECORD_READ_JSON} ${check_source})
  set_tests_properties(${exe}.rdc PROPERTIES LABELS "MnemeReplay::RDC")
endfunction()


find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(RECORD_READ_JSON "${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/run-replay.py")

set(MNEME_BIN_FILE ${CMAKE_BINARY_DIR}/src/replay)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run-replay.py ${CMAKE_CURRENT_BINARY_DIR}/run-replay.py @ONLY)



file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py "
import lit.formats

config.name = 'LIT tests'
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.cpp']
config.test_source_root = '${CMAKE_CURRENT_SOURCE_DIR}'
config.test_exec_root = '${CMAKE_CURRENT_BINARY_DIR}'
MNEME_PRELOAD_LIB='${CMAKE_BINARY_DIR}/src/librecord.so'
PAGE_SIZE = lit_config.params['PG']
RR = lit_config.params['RR']
EXT = lit_config.params['ext']
FILECHECK = lit_config.params['FILECHECK']
config.substitutions.append(('MNEME_PRELOAD_LIB', MNEME_PRELOAD_LIB))
config.substitutions.append(('%PG', PAGE_SIZE))
config.substitutions.append(('%RR', RR))
config.substitutions.append(('%ext', EXT))
config.substitutions.append(('%FILECHECK', FILECHECK))
"
)

CREATE_GPU_TEST(test_rr_global test_global.cpp)
CREATE_RDC_GPU_TEST(test_rr_global test_global.cpp)

CREATE_GPU_TEST(test_rr_device_mem test_device_mem.cpp)
CREATE_RDC_GPU_TEST(test_rr_device_mem test_device_mem.cpp)

