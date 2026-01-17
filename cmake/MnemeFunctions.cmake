# Copyright (c) 2021-2024. LLNL-CODE-2000766 and Mneme Contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

function(add_mneme target)
    # Link proteus as a shared library for preloading.
    # TODO: Change to static linking and use linker wrapper flags for interposing.
    add_proteus(${target} FORCE_JIT_ANNOTATE_ALL LINK_SHARED)
    string(TOLOWER "${CMAKE_CXX_COMPILER}" _cxx)
    set(_looks_like_clang FALSE)
    if(_cxx MATCHES "clang|amdclang|hipcc")
      set(_looks_like_clang TRUE)
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR _looks_like_clang)
      target_compile_options(${target}
        PRIVATE
          -fno-discard-value-names
          -ftrivial-auto-var-init=zero
      )
    endif()


endfunction()
