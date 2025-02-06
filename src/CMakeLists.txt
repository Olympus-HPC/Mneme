
set(RecordLibSRC "")
set(ReplaySRC MnemeReplay.cpp MnemePageManager.cpp MnemeDeviceCode.cpp)
if (MNEME_ENABLE_HIP)
  list(APPEND RecordLibSRC MnemeRecordPreloadHIP.cpp MnemePageManager.cpp)
endif()

if (MNEME_ENABLE_LOGGER)
  list(APPEND RecordLibSRC MnemeLogger.cpp)
  list(APPEND ReplaySRC MnemeLogger.cpp)
endif()

add_library(record SHARED ${RecordLibSRC})
add_executable(replay ${ReplaySRC})

if (MNEME_ENABLE_HIP)
  set_source_files_properties(MnemeDeviceCode.cpp PROPERTIES LANGUAGE HIP)
endif()


target_include_directories(record SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
target_include_directories(replay SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(replay PRIVATE proteus)

if (MNEME_ENABLE_LOGGER)
target_link_libraries(replay PRIVATE spdlog::spdlog_header_only)
target_compile_definitions(replay PRIVATE MNEME_ENABLE_LOGGING)
endif()

if(MNEME_ENABLE_HIP)
  target_link_libraries(record PRIVATE hip::host)
  target_link_libraries(record PRIVATE ${LLVM_AVAILABLE_LIBS})
  set_target_properties(record PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

  target_link_libraries(replay PRIVATE hip::host)
  target_link_libraries(replay PRIVATE ${LLVM_AVAILABLE_LIBS})
  set_target_properties(replay PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

get_target_property(PROTEUS_LIB proteus LOCATION)
message(STATUS "Proteus library is expected at: ${PROTEUS_LIB}")

if (MNEME_ENABLE_LOGGER)
  target_compile_definitions(record PRIVATE MNEME_ENABLE_LOGGER)
  target_link_libraries(record PRIVATE spdlog::spdlog_header_only)
endif()

install(
  TARGETS record replay
  EXPORT mnemeTargets
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  )
