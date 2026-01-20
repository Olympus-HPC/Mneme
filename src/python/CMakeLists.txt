add_subdirectory(profile)

set(LLVM_SRC assembly.cpp bitcode.cpp core.cpp initfini.cpp
            module.cpp value.cpp executionengine.cpp type.cpp attribute.cpp
            targets.cpp)

file(GLOB LLVM_SRC "llvm/*.cpp")
file(GLOB PROTEUS_SRC "proteus/*.cpp")
file(GLOB MNEME_SRC "./*.cpp")
file(GLOB TRANSFORMS "transforms/*.cpp")


add_library(mneme SHARED ${LLVM_SRC} ${PROTEUS_SRC} ${MNEME_SRC} ${TRANSFORMS} ${MnemeCoreSrcDir}/MnemePageManager.cpp ${MnemeCoreSrcDir}/MnemeDeviceCode.cpp)

if (MNEME_ENABLE_HIP)
  set_source_files_properties(${MnemeCoreSrcDir}/MnemeDeviceCode.cpp PROPERTIES LANGUAGE HIP)
elseif(MNEME_ENABLE_CUDA)
  set_source_files_properties(${MnemeCoreSrcDir}/MnemeDeviceCode.cpp PROPERTIES LANGUAGE CUDA)
endif()


target_include_directories(mneme SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
# FIXME: Here now we need to be "careful". If LLVM brings in the bundled .so we need to 
# link in proteusCore and be done with it.
target_link_libraries(mneme PRIVATE proteusCore)

if (MNEME_ENABLE_LOGGER)
target_link_libraries(mneme PRIVATE spdlog::spdlog_header_only)
target_compile_definitions(mneme PRIVATE MNEME_ENABLE_LOGGER)
endif()


target_link_libraries(mneme PRIVATE ${LLVM_AVAILABLE_LIBS})
