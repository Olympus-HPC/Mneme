
set(MnemeCoreSrcDir ${CMAKE_CURRENT_SOURCE_DIR})
set(RecordLibSRC "")
set(ReplaySRC MnemeReplay.cpp MnemePageManager.cpp MnemeDeviceCode.cpp MnemeSymbols.cpp)
if (MNEME_ENABLE_HIP)
  list(APPEND RecordLibSRC MnemeRecordPreloadHIP.cpp MnemePageManager.cpp MnemeSymbols.cpp)
elseif(MNEME_ENABLE_CUDA)
	list(APPEND RecordLibSRC MnemeRecordPreloadCUDA.cpp MnemePageManager.cpp MnemeSymbols.cpp MnemeCUDABinary.cpp)
endif()

add_library(record SHARED ${RecordLibSRC})
add_library(mneme_shallow SHARED MnemeShallowRuntime.cpp)
add_executable(replay ${ReplaySRC})


if(MNEME_LINK_SHARED_LLVM)
  if(LLVM_LINK_LLVM_DYLIB)
    llvm_config(record USE_SHARED)
    llvm_config(replay USE_SHARED)
  else()
    find_library(LLVM_SHARED_LIB
      NAMES LLVM
      PATHS "${LLVM_LIBRARY_DIR}"
      NO_DEFAULT_PATH
    )
    if(LLVM_SHARED_LIB)
      message(STATUS "Found libLLVM.so: ${LLVM_SHARED_LIB}")
      set(llvm_libs ${LLVM_SHARED_LIB})
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
  set(llvm_libs ${LLVM_AVAILABLE_LIBS})
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
target_include_directories(replay SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(replay PRIVATE proteusCore)

if (MNEME_ENABLE_LOGGER)
target_link_libraries(replay PRIVATE spdlog::spdlog_header_only)
target_link_libraries(record PRIVATE spdlog::spdlog_header_only)
target_compile_definitions(replay PRIVATE MNEME_ENABLE_LOGGER)
target_compile_definitions(record PRIVATE MNEME_ENABLE_LOGGER)
endif()

if(MNEME_ENABLE_HIP)
target_link_libraries(record PRIVATE hip::host)
target_link_libraries(record PRIVATE ${llvm_libs})
set_target_properties(record PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

target_include_directories(mneme_shallow PRIVATE ${HIP_INCLUDE_DIRS})

target_link_libraries(replay PRIVATE hip::host)
target_link_libraries(replay PRIVATE ${llvm_libs})

target_link_libraries(replay PRIVATE ${llvm_libs})
set_target_properties(replay PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
elseif(MNEME_ENABLE_CUDA)
target_link_libraries(record PRIVATE CUDA::cudart CUDA::cuda_driver)
target_link_libraries(record PRIVATE ${llvm_libs})
set_target_properties(record PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)


target_include_directories(mneme_shallow PRIVATE ${CUDA_INCLUDE_DIRS})

target_link_libraries(replay PRIVATE CUDA::cudart CUDA::cuda_driver)
target_link_libraries(replay PRIVATE ${llvm_libs})

target_link_libraries(replay PRIVATE ${llvm_libs})
set_target_properties(replay PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()


if (MNEME_ENABLE_AUTOTUNE)
  add_subdirectory(python)
endif()
