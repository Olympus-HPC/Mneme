# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MNEMELib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.24)
cmake_policy(SET CMP0104 NEW)

# Define the project
project(MNEMEExamples LANGUAGES CXX)


option(WITH_MNEME_EXAMPLE_CUDA  "Option to enable CUDA" Off)
option(WITH_MNEME_EXAMPLE_HIP "Option to enable HIP" Off)

find_package(mneme REQUIRED)
find_package(proteus REQUIRED)

if(WITH_MNEME_EXAMPLE_CUDA AND WITH_MNEME_EXAMPLE_HIP)
  message(FATAL_ERROR
    "WITH_MNEME_EXAMPLE_CUDA=${WITH_MNEME_EXAMPLE_CUDA} and "
    "WITH_MNEME_EXAMPLE_HIP=${WITH_MNEME_EXAMPLE_HIP} are mutually exclusive."
  )
endif()


set(MNEME_INPUT_DIR "${CMAKE_BINARY_DIR}/inputs")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (WITH_MNEME_EXAMPLE_CUDA)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)
  enable_language(CUDA)
  find_package(CUDAToolkit)
elseif (WITH_MNEME_EXAMPLE_HIP)
  enable_language(HIP)
  find_package(hip REQUIRED)
endif()

enable_testing()

add_subdirectory(wsm5)
add_subdirectory(miniFE)
add_subdirectory(bezier-surface)
