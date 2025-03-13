# Locate Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)


add_subdirectory(proteus)
