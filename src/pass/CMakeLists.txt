add_library(MnemePassPlugin MODULE MnemePurgeAutoInitPass.cpp)

target_include_directories(MnemePassPlugin SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})

if(MNEME_LINK_SHARED_LLVM AND LLVM_LINK_LLVM_DYLIB)
  llvm_config(MnemePassPlugin USE_SHARED)
endif()
target_link_libraries(MnemePassPlugin PRIVATE ${MNEME_LLVM_LIBS})

target_link_options(MnemePassPlugin PRIVATE
  "$<$<PLATFORM_ID:Darwin>:LINKER:SHELL:-undefined dynamic_lookup>"
  "$<$<AND:$<NOT:$<PLATFORM_ID:Darwin>>,$<BOOL:${UNIX}>>:-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/MnemePassPlugin.exports.map>")
set_target_properties(MnemePassPlugin PROPERTIES
  LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/MnemePassPlugin.exports.map")
