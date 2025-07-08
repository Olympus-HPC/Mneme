find_program(LIT lit
    PATHS ${LLVM_INSTALL_DIR}/bin)

if(NOT LIT)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import lit"
        RESULT_VARIABLE EXIT_CODE
        OUTPUT_QUIET
    )

    if(${EXIT_CODE} EQUAL 0)
        set(LIT ${CMAKE_CURRENT_SOURCE_DIR}/lit-main.py)
        message(STATUS "Using ${LIT} for lit testing")
    else()
        message(FATAL_ERROR "Testing requires lit to be installed, e.g., python -m pip install lit>=15.0.7")
    endif()
endif()
message(STATUS "Found lit at ${LIT}")


if(MNEME_ENABLE_HIP)
    enable_language(HIP)
elseif(MNEME_ENABLE_CUDA)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        message(FATAL_ERROR "Set CMAKE_CUDA_ARCHITECTURES to compile for")
    endif()

    enable_language(CUDA)
    message(STATUS "CUDA compiler ${CMAKE_CUDA_COMPILER_ID}")

    if(NOT ${CMAKE_CUDA_COMPILER_ID} STREQUAL "Clang")
        message(FATAL_ERROR "JIT is compatible only with Clang CUDA compilation")
    endif()
endif()


find_program(FILECHECK FileCheck
    PATHS ${LLVM_INSTALL_DIR}/bin ${LLVM_INSTALL_DIR}/libexec/llvm)

if(NOT FILECHECK)
    message(FATAL_ERROR "Testing requires FileCheck to be installed")
else()
    message(STATUS "Found Filecheck at ${FILECHECK}")
endif()


add_subdirectory(unit_tests)
#add_subdirectory(memory)
add_subdirectory(record)
#add_subdirectory(replay)
