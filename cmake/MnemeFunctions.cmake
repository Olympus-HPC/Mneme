function(add_mneme target)
    add_proteus(${target} FORCE_JIT_ANNOTATE_ALL)
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
