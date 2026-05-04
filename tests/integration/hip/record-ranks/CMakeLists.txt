cmake_minimum_required(VERSION 3.21)

project(record-ranks LANGUAGES HIP CXX C)

find_package(hip REQUIRED CONFIG)
find_package(MPI REQUIRED)
find_package(mneme REQUIRED CONFIG HINTS ${mneme_DIR})

add_executable(record_ranks record_ranks.cpp)
set_source_files_properties(record_ranks.cpp PROPERTIES LANGUAGE HIP)
target_link_libraries(record_ranks PRIVATE MPI::MPI_CXX hip::host)
target_compile_definitions(record_ranks PRIVATE MNEME_ENABLE_HIP=1)

add_mneme(record_ranks)
