include(ExternalProject)

ExternalProject_Add(
   proteus  
    GIT_REPOSITORY git@github.com:Olympus-HPC/proteus.git
    GIT_TAG        features/mneme-integration
    SOURCE_DIR     ${CMAKE_BINARY_DIR}/_deps/MyDependency-src
    BINARY_DIR     ${CMAKE_BINARY_DIR}/_deps/MyDependency-build
    INSTALL_DIR    ${CMAKE_INSTALL_PREFIX}

    # Custom CMake configuration
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DLLVM_INSTALL_DIR=${LLVM_INSTALL_DIR} 
        -DCMAKE_C_COMPILER=${LLVM_INSTALL_DIR}/bin/clang
        -DCMAKE_CXX_COMPILER=${LLVM_INSTALL_DIR}/bin/clang++
        -DENABLE_HIP=${ENABLE_HIP}
        -DENABLE_CUDA=${ENABLE_CUDA}
        -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
        -DENABLE_TESTS=Off

    # Custom build step
    BUILD_COMMAND make -j ${CMAKE_BUILD_PARALLEL_LEVEL}

    # Custom install step
    INSTALL_COMMAND make install
)
