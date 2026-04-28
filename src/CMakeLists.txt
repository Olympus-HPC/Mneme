
set(MnemeCoreSrcDir ${CMAKE_CURRENT_SOURCE_DIR})
set(AnnotationRuntimeSRC MnemeAnnotation.cpp)
set(RecordLibSRC MnemeAnnotationInternal.cpp)
set(ReplaySRC MnemeReplay.cpp MnemePageManager.cpp MnemeDeviceCode.cpp MnemeAnnotationInternal.cpp)

if (MNEME_ENABLE_HIP)
  list(APPEND RecordLibSRC MnemeRecordPreloadHIP.cpp MnemePageManager.cpp)
elseif(MNEME_ENABLE_CUDA)
  list(APPEND RecordLibSRC MnemeRecordPreloadCUDA.cpp MnemePageManager.cpp)
endif()

add_library(mnemert SHARED ${AnnotationRuntimeSRC})
target_include_directories(
 mnemert 
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

add_library(record SHARED ${RecordLibSRC})

if(MNEME_LINK_SHARED_LLVM)
  if(LLVM_LINK_LLVM_DYLIB)
    llvm_config(record USE_SHARED)
  else()
    find_library(LLVM_SHARED_LIB
      NAMES LLVM
      PATHS "${LLVM_LIBRARY_DIR}"
      NO_DEFAULT_PATH
    )
    if(LLVM_SHARED_LIB)
      message(STATUS "Found libLLVM.so: ${LLVM_SHARED_LIB}")
      set(llvm_libs ${LLVM_AVAILABLE_LIBS} CACHE INTERNAL "Points to the actual LLVM LIBS to be linked with")
    else()
      message(FATAL_ERROR "The LLVM installation at ${LLVM_DIR} does not provide libLLVM.so, required due to "
      "MNEME_LINK_SHARED_LLVM=ON.
      Change MNEME_LINK_SHARED_LLVM to OFF.
      ")
    endif()
  endif()
else()
  if(LLVM_LINK_LLVM_DYLIB)
    message(FATAL_ERROR "The LLVM installation at ${LLVM_DIR} requires to link with "
    "the LLVM shared library, but MNEME_LINK_SHARED_LLVM=OFF.
    Change MNEME_LINK_SHARED_LLVM to ON.")
  endif()
  set(llvm_libs ${LLVM_AVAILABLE_LIBS} CACHE INTERNAL "Points to the actual LLVM LIBS to be linked with")
  # Some LLVM installations include in the LLVM_AVAILABLE_LIBS both an LLVM
  # shared library AND static libraries. The LLVM shared library bundles those
  # static libraries and we have to remove it to avoid static initialization
  # errors from linking with both.
  list(REMOVE_ITEM llvm_libs "LLVM")
endif()

message(STATUS "LLVM Libraries are: ${llvm_libs}")

if (MNEME_ENABLE_HIP)
  set_source_files_properties(MnemeDeviceCode.cpp PROPERTIES LANGUAGE HIP)
elseif(MNEME_ENABLE_CUDA)
  set_source_files_properties(MnemeDeviceCode.cpp PROPERTIES LANGUAGE CUDA)
endif()


target_include_directories(record SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(record PRIVATE mnemert)
# Link the record preload shared library with proteus shared library to avoid
# LLVM static initialization issues.
target_link_libraries(record PRIVATE proteus)

if (MNEME_ENABLE_LOGGER)
  message(STATUS "Adding SPDLOG in recording")
  target_link_libraries(mnemert PRIVATE spdlog::spdlog_header_only)
  target_compile_definitions(mnemert PRIVATE MNEME_ENABLE_LOGGER)
  target_link_libraries(record PRIVATE spdlog::spdlog_header_only)
  target_compile_definitions(record PRIVATE MNEME_ENABLE_LOGGER)
endif()

if(MNEME_ENABLE_HIP)
  target_link_libraries(record PRIVATE hip::host)
  target_link_libraries(record PRIVATE ${llvm_libs})
  # Proteus's HIP impl headers (CoreLLVMHIP.h) call lld::elf::link inline,
  # so consumers that instantiate that code must link the LLD libraries
  # themselves — libproteus.so hides them via its version script.
  target_link_libraries(record PRIVATE lldCommon lldELF)
elseif(MNEME_ENABLE_CUDA)
  target_link_libraries(record PRIVATE CUDA::cudart CUDA::cuda_driver)
  target_link_libraries(record PRIVATE ${llvm_libs})
endif()

set_target_properties(record PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

if (MNEME_ENABLE_AUTOTUNE)
  add_subdirectory(python)
endif()
