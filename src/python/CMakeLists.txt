set(LLVM_SRC assembly.cpp bitcode.cpp core.cpp initfini.cpp
            module.cpp value.cpp executionengine.cpp type.cpp attribute.cpp
            targets.cpp)

file(GLOB LLVM_SRC "llvm/*.cpp")
file(GLOB PROTEUS_SRC "proteus/*.cpp")
file(GLOB MNEME_SRC "./*.cpp")


add_library(mneme SHARED ${LLVM_SRC} ${PROTEUS_SRC} ${MNEME_SRC})
target_include_directories(mneme SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mneme PRIVATE proteusCore)

if (MNEME_ENABLE_LOGGER)
target_link_libraries(mneme PRIVATE spdlog::spdlog_header_only)
target_compile_definitions(mneme PRIVATE MNEME_ENABLE_LOGGER)
endif()


target_link_libraries(mneme PRIVATE ${LLVM_AVAILABLE_LIBS})
