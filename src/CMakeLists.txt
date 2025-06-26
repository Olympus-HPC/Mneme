
set(MnemeCoreSrcDir ${CMAKE_CURRENT_SOURCE_DIR})
set(RecordLibSRC "")
set(ReplaySRC MnemeReplay.cpp MnemePageManager.cpp MnemeDeviceCode.cpp MnemeSymbols.cpp)
if (MNEME_ENABLE_HIP)
  list(APPEND RecordLibSRC MnemeRecordPreloadHIP.cpp MnemePageManager.cpp MnemeSymbols.cpp)
elseif(MNEME_ENABLE_CUDA)
  list(APPEND RecordLibSRC MnemePageManager.cpp MnemeSymbols.cpp)
endif()

add_library(record SHARED ${RecordLibSRC})
add_executable(replay ${ReplaySRC})

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
  target_link_libraries(record PRIVATE ${LLVM_AVAILABLE_LIBS})
  set_target_properties(record PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

  target_link_libraries(replay PRIVATE hip::host)
  target_link_libraries(replay PRIVATE ${LLVM_AVAILABLE_LIBS})

  #target_link_libraries(replay PRIVATE ${LLVM_AVAILABLE_LIBS})
  set_target_properties(replay PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
elseif(MNEME_ENABLE_CUDA)
  target_link_libraries(record PRIVATE CUDA::cudart)
  target_link_libraries(record PRIVATE ${LLVM_AVAILABLE_LIBS})
  set_target_properties(record PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

  target_link_libraries(replay PRIVATE CUDA::cudart)
  target_link_libraries(replay PRIVATE ${LLVM_AVAILABLE_LIBS})

  #target_link_libraries(replay PRIVATE ${LLVM_AVAILABLE_LIBS})
  set_target_properties(replay PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

endif()


if (MNEME_ENABLE_AUTOTUNE)
  add_subdirectory(python)
endif()
