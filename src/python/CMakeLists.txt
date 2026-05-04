add_subdirectory(profile)

set(LLVM_SRC assembly.cpp bitcode.cpp core.cpp initfini.cpp
            module.cpp value.cpp executionengine.cpp type.cpp attribute.cpp
            targets.cpp)

file(GLOB LLVM_SRC "llvm/*.cpp")
file(GLOB PROTEUS_SRC "proteus/*.cpp")
file(GLOB MNEME_SRC "./*.cpp")
file(GLOB TRANSFORMS "transforms/*.cpp")


add_library(mneme SHARED ${LLVM_SRC} ${PROTEUS_SRC} ${MNEME_SRC} ${TRANSFORMS}
                  ${MnemeCoreSrcDir}/MnemePageManager.cpp
                  ${MnemeCoreSrcDir}/MnemeDeviceCode.cpp
                  ${MnemeCoreSrcDir}/MnemeAnnotationInternal.cpp
                  ${MnemeCoreSrcDir}/MnemeRank.cpp)
message(STATUS "LLVM LIBS are found to be : ${llvm_libs}")

if (MNEME_ENABLE_HIP)
  set_source_files_properties(${MnemeCoreSrcDir}/MnemeDeviceCode.cpp PROPERTIES LANGUAGE HIP)
  set_target_properties(mneme PROPERTIES LINKER_LANGUAGE HIP)
  target_compile_definitions(mneme PRIVATE MNEME_ENABLE_HIP)
elseif(MNEME_ENABLE_CUDA)
  set_source_files_properties(${MnemeCoreSrcDir}/MnemeDeviceCode.cpp PROPERTIES LANGUAGE CUDA)
  target_compile_definitions(mneme PRIVATE MNEME_ENABLE_CUDA)
endif()

if(MNEME_LINK_SHARED_LLVM)
  message(STATUS "MNEMNE_LINK_SHARED_LLVM is On ${MNEME_LINK_SHARED_LLVM}")
  if(LLVM_LINK_LLVM_DYLIB)
    llvm_config(mneme USE_SHARED)
  else()
    find_library(LLVM_SHARED_LIB
      NAMES LLVM
      PATHS "${LLVM_LIBRARY_DIR}"
      NO_DEFAULT_PATH
    )
    if(LLVM_SHARED_LIB)
      message(STATUS "Found libLLVM.so: ${LLVM_SHARED_LIB}")
      target_link_libraries(mneme PRIVATE ${LLVM_AVAILABLE_LIBS})
    else()
      message(FATAL_ERROR "The LLVM installation at ${LLVM_DIR} does not provide libLLVM.so, required due to "
      "MNEME_LINK_SHARED_LLVM=ON.
      Change MNEME_LINK_SHARED_LLVM to OFF.
      ")
    endif()
  endif()
else()
  message(STATUS "MNEMNE_LINK_SHARED_LLVM is Off")
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
  target_link_libraries(mneme PRIVATE ${llvm_libs})
endif()


target_include_directories(mneme SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
# FIXME: Here now we need to be "careful". If LLVM brings in the bundled .so we need to 
# link in proteusCore and be done with it.
target_link_libraries(mneme PRIVATE proteusCore)

if (MNEME_ENABLE_LOGGER)
  target_link_libraries(mneme PRIVATE spdlog::spdlog_header_only)
  target_compile_definitions(mneme PRIVATE MNEME_ENABLE_LOGGER)
endif()

if(MNEME_ENABLE_HIP)
  # Proteus's HIP impl headers (CoreLLVMHIP.h) call lld::elf::link inline,
  # so consumers that instantiate that code must link the LLD libraries
  # themselves — libproteus.so hides them via its version script.
  target_link_libraries(mneme PRIVATE hip::host lldCommon lldELF)
elseif(MNEME_ENABLE_CUDA)
  target_link_libraries(mneme PRIVATE CUDA::cudart CUDA::cuda_driver)
  if(MNEME_NVPTXCOMPILER_LIBRARY MATCHES "\\.a$")
    get_filename_component(_mneme_nvptxcompiler_name "${MNEME_NVPTXCOMPILER_LIBRARY}" NAME)
    target_link_libraries(mneme PRIVATE
      -Wl,--whole-archive
      "${MNEME_NVPTXCOMPILER_LIBRARY}"
      -Wl,--no-whole-archive
      "-Wl,--exclude-libs,${_mneme_nvptxcompiler_name}")
  else()
    target_link_libraries(mneme PRIVATE "${MNEME_NVPTXCOMPILER_LIBRARY}")
  endif()
endif()


set_target_properties(mneme PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
