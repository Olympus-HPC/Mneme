
set(RecordLibSRC "")
set(ReplaySRC MnemeReplay.cpp MnemePageManager.cpp MnemeDeviceCode.cpp)
if (MNEME_ENABLE_HIP)
  list(APPEND RecordLibSRC MnemeRecordPreloadHIP.cpp MnemePageManager.cpp)
endif()

if (MNEME_ENABLE_LOGGER)
  list(APPEND RecordLibSRC MnemeLogger.cpp)
  list(APPEND ReplaySRC MnemeLogger.cpp)
endif()

add_library(mnemerecord SHARED ${RecordLibSRC})
add_executable(mnemereplay ${ReplaySRC})

if (MNEME_ENABLE_HIP)
  set_source_files_properties(MnemeDeviceCode.cpp PROPERTIES LANGUAGE HIP)
endif()


target_include_directories(mnemerecord SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
target_include_directories(mnemereplay SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mnemereplay PRIVATE proteus)

if (MNEME_ENABLE_LOGGER)
target_link_libraries(mnemereplay PRIVATE spdlog::spdlog_header_only)
target_compile_definitions(mnemereplay PRIVATE MNEME_ENABLE_LOGGING)
endif()

if(MNEME_ENABLE_HIP)
  target_link_libraries(mnemerecord PRIVATE hip::host)
  target_link_libraries(mnemerecord PRIVATE ${LLVM_AVAILABLE_LIBS})
  set_target_properties(mnemerecord PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

  target_link_libraries(mnemereplay PRIVATE hip::host)
  target_link_libraries(mnemereplay PRIVATE ${LLVM_AVAILABLE_LIBS})
  set_target_properties(mnemereplay PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

get_target_property(PROTEUS_LIB proteus LOCATION)
message(STATUS "Proteus library is expected at: ${PROTEUS_LIB}")

add_library(RR::MnemeRecord ALIAS mnemerecord)
if (MNEME_ENABLE_LOGGER)
  target_compile_definitions(mnemerecord PRIVATE MNEME_ENABLE_LOGGER)
  target_link_libraries(mnemerecord PRIVATE spdlog::spdlog_header_only)
endif()

install(
  TARGETS mnemerecord mnemereplay
  EXPORT mnemeTargets
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  )
