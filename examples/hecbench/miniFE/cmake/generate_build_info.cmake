# Inputs (passed via -D...):
#  OUT, TEMPLATE, HEADER_PREFIX, MACRO_PREFIX, CXX, CXXFLAGS, MODE

if(NOT DEFINED OUT OR NOT DEFINED TEMPLATE OR NOT DEFINED HEADER_PREFIX OR NOT DEFINED MACRO_PREFIX)
  message(FATAL_ERROR "Missing required -DOUT/-DTEMPLATE/-DHEADER_PREFIX/-DMACRO_PREFIX")
endif()

if(NOT DEFINED MODE)
  set(MODE "repro") # repro|host
endif()

# Compiler path
set(CXX_PATH "unknown")
if(DEFINED CXX AND NOT "${CXX}" STREQUAL "")
  get_filename_component(_cxx_resolved "${CXX}" REALPATH BASE_DIR "${CMAKE_BINARY_DIR}" )
  if(EXISTS "${_cxx_resolved}")
    set(CXX_PATH "${_cxx_resolved}")
  else()
    # try to locate in PATH
    find_program(_cxx_prog NAMES "${CXX}")
    if(_cxx_prog)
      set(CXX_PATH "${_cxx_prog}")
    endif()
  endif()
endif()

# Compiler version (first non-empty line)
set(CXX_VERSION "unknown")
if(DEFINED CXX AND NOT "${CXX}" STREQUAL "")
  execute_process(COMMAND "${CXX}" --version
                  OUTPUT_VARIABLE _ver OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_VARIABLE _ver_err ERROR_STRIP_TRAILING_WHITESPACE
                  RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    execute_process(COMMAND "${CXX}" -V
                    OUTPUT_VARIABLE _ver OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_VARIABLE _ver_err ERROR_STRIP_TRAILING_WHITESPACE
                    RESULT_VARIABLE _rc2)
  endif()
  string(REPLACE "\r\n" "\n" _ver "${_ver}")
  string(REPLACE "\r" "\n" _ver "${_ver}")
  string(REGEX REPLACE "\n.*" "" CXX_VERSION "${_ver}")
  if("${CXX_VERSION}" STREQUAL "")
    set(CXX_VERSION "unknown")
  endif()
endif()

# Host info (optional for reproducibility)
set(HOSTNAME "unknown")
set(KERNEL_NAME "unknown")
set(KERNEL_RELEASE "unknown")
set(PROCESSOR "unknown")

if(MODE STREQUAL "host")
  execute_process(COMMAND ${CMAKE_COMMAND} -E hostname OUTPUT_VARIABLE HOSTNAME OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND uname -s OUTPUT_VARIABLE KERNEL_NAME OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _u1)
  execute_process(COMMAND uname -r OUTPUT_VARIABLE KERNEL_RELEASE OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _u2)
  execute_process(COMMAND uname -p OUTPUT_VARIABLE PROCESSOR OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _u3)
endif()

if(NOT DEFINED CXXFLAGS)
  set(CXXFLAGS "")
endif()

# Set variables used by configure_file
set(HEADER_PREFIX "${HEADER_PREFIX}")
set(MACRO_PREFIX "${MACRO_PREFIX}")

configure_file("${TEMPLATE}" "${OUT}" @ONLY)

