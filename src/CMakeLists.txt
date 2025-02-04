
set(RecordLibSRC "")
set(ReplaySRC MnemeReplay.cpp MnemePageManager.cpp MnemeDeviceCode.cpp)
if (ENABLE_HIP)
  list(APPEND RecordLibSRC MnemeRecordPreloadHIP.cpp MnemePageManager.cpp)
endif()

if (ENABLE_LOGGER)
  list(APPEND RecordLibSRC MnemeLogger.cpp)
  list(APPEND ReplaySRC MnemeLogger.cpp)
endif()

add_library(mnemerecord SHARED ${RecordLibSRC})
add_executable(mnemereplay ${ReplaySRC})

if (ENABLE_HIP)
  set_source_files_properties(MnemeDeviceCode.cpp PROPERTIES LANGUAGE HIP)
endif()


target_include_directories(mnemerecord SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/common)
target_include_directories(mnemereplay SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/common)
target_link_libraries(mnemereplay PRIVATE proteus)
get_target_property(LIBRARY_PATH proteus IMPORTED_LOCATION)
get_filename_component(LIBRARY_DIR ${LIBRARY_PATH} DIRECTORY)
message(STATUS "Library directory: ${LIBRARY_DIR}")

if (ENABLE_LOGGER)
target_link_libraries(mnemereplay PRIVATE spdlog::spdlog_header_only)
target_compile_definitions(mnemereplay PRIVATE MNEME_ENABLE_LOGGING)
endif()

if(ENABLE_HIP)
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
if (ENABLE_LOGGER)
target_compile_definitions(mnemerecord PRIVATE MNEME_ENABLE_LOGGING)
target_link_libraries(mnemerecord PRIVATE spdlog::spdlog_header_only)
endif()

#target_include_directories(replay PRIVATE ${LLVM_INCLUDE_DIRS})

#set_target_properties(replay PROPERTIES
#  INSTALL_RPATH "${LLVM_LIBRARY_DIR}")

#install(TARGETS replay)

#install(TARGETS record EXPORT RR 
#    LIBRARY DESTINATION lib
#    ARCHIVE DESTINATION lib
#    RUNTIME DESTINATION bin
#    INCLUDES DESTINATION include)

# Install the export set for the library
#install(EXPORT RR 
#  FILE RRTargets.cmake
#  NAMESPACE RR::
#  DESTINATION lib/cmake/RR)

# Create a version file for compatibility checks
#write_basic_package_version_file(
#  "${CMAKE_CURRENT_BINARY_DIR}/RRConfigVersion.cmake"
#    VERSION ${PROJECT_VERSION}
#    COMPATIBILITY AnyNewerVersion)
#
## Configure the config file for the project
#configure_package_config_file(
#  "${CMAKE_SOURCE_DIR}/cmake/RRConfig.cmake.in"
#  "${CMAKE_CURRENT_BINARY_DIR}/RRConfig.cmake"
#  INSTALL_DESTINATION lib/cmake/RR)
#
## Install the config files
#install(FILES
#  "${CMAKE_CURRENT_BINARY_DIR}/RRConfig.cmake"
#  "${CMAKE_CURRENT_BINARY_DIR}/RRConfigVersion.cmake"
#  DESTINATION lib/cmake/RR)
