
set(RecordLibSRC "")
if (ENABLE_HIP)
  list(APPEND RecordLibSRC MnemeRecordPreloadHIP.cpp MnemePageManager.cpp)
endif()

add_library(mnemerecord SHARED ${RecordLibSRC})
add_executable(mnemereplay MnemeReplay.cpp MnemePageManager.cpp MnemeDeviceCode.cpp)

if (ENABLE_HIP)
  set_source_files_properties(MnemeDeviceCode.cpp PROPERTIES LANGUAGE HIP)
endif()

#add_executable(replay replay.cpp memory.cpp jit.cpp common.cpp)

set(libs ${LLVM_AVAILABLE_LIBS})


target_include_directories(mnemerecord SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/common)
target_include_directories(mnemereplay SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/common)
target_link_libraries(mnemereplay PRIVATE proteus)
if(ENABLE_HIP)
  target_link_libraries(mnemerecord PRIVATE hip::host)
  target_link_libraries(mnemereplay PRIVATE ${LLVM_AVAILABLE_LIBS})

  target_link_libraries(mnemerecord PRIVATE ${LLVM_AVAILABLE_LIBS})
  target_link_libraries(mnemereplay PRIVATE hip::host)
endif()


add_library(RR::MnemeRecord ALIAS mnemerecord)

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
