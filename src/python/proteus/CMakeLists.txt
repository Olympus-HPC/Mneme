
# Define the Python module name
# Function to remove -fno-rtti from target compile options

pybind11_add_module(coreJIT ProteusBindings.cpp)

# Link against Proteus (ensure Proteus is built and available)
target_link_libraries(coreJIT PRIVATE proteusCore)

# Set output directory so the module ends up in the correct Python package
set_target_properties(coreJIT PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python)

# Ensure RTTI is enabled explicitly
target_compile_options(coreJIT PRIVATE -frtti)

