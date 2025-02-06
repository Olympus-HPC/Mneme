
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
get_target_property(LIBRARY_PATH proteus IMPORTED_LOCATION)
get_filename_component(LIBRARY_DIR ${LIBRARY_PATH} DIRECTORY)
message(STATUS "Library directory: ${LIBRARY_DIR}")

if (MNEME_ENABLE_LOGGER)
target_link_libraries(mnemereplay PRIVATE spdlog::spdlog_header_only)
target_compile_definitions(mnemereplay PRIVATE MNEME_ENABLE_LOGGING)
endif()

if(MNEME_ENABLE_HIP)
  target_link_libraries(mnemerecord PRIVATE hip::host)
  target_link_libraries(mnemerecord PRIVATE ${LLVM_AVAILABLE_LIBS})
  target_link_directories(mnemerecord PRIVATE ${LLVM_LIBRARY_DIR})
  set_target_properties(mnemerecord PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

  target_link_libraries(mnemereplay PRIVATE hip::host)
  target_link_libraries(mnemereplay PRIVATE ${LLVM_AVAILABLE_LIBS})
  target_link_directories(mnemereplay PRIVATE ${LLVM_LIBRARY_DIR})
  set_target_properties(mnemereplay PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()
message(STATUS "LLVM LIBS: ${LLVM_AVAILABLE_LIBS}")
message(STATUS "LLVM DIRECTORY: ${LLVM_LIBRARY_DIR}")


add_library(RR::MnemeRecord ALIAS mnemerecord)
if (MNEME_ENABLE_LOGGER)
  target_compile_definitions(mnemerecord PRIVATE MNEME_ENABLE_LOGGER)
target_link_libraries(mnemerecord PRIVATE spdlog::spdlog_header_only)
endif()

