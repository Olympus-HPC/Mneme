# Installation

Currently, Mneme is distributed using its git repo to build and install it.
We recommend using the **latest develop** branch version, which is well tested and
robust while including the most recent features.

Mneme builds and installs three components: the LLVM plugin pass, the Recording library,
and the Mneme replay tool.
The user must integrate only the LLVM pass in their build system and use the recording library when running executables.
We provide information on how to integrate Mneme with your application in the
[Integration](integration.md) section.

## Building


### Dependencies

Mneme depends on LLVM (ROCm@6.2 or ROCm@6.3) on [Proteus](https://github.com/Olympus-HPC/proteus/). 

The project uses `cmake` for building and depends on an LLVM installation (CI
tests cover 18 and AMD ROCm versions 6.2.1, 6.3).  The top-level
`CMakeLists.txt` has the following (binary) build options:

* `MNEME_ENABLE_TESTS`: builds Mneme tests.
* `MNEME_ENABLE_HIP`: enables HIP support.
* `MNEME_ENABLE_CUDA`: enable CUDA support (**support for cuda is work in progress**).
* `MNEME_ENABLE_DEBUG`: logs debugging information (for developers).
* `MNEME_ENABLE_LOGGER`: Enalbes Mneme logging.


Please check the buld process of our CI [here](https://github.com/Olympus-HPC/Mneme/blob/features/amd-refactor/scripts/gitlab/ci-build-test.sh). 

The script clones proteus and spdlog builds and installs them and then installs Mneme.

## Testing

It is advised to enable tests when deploying Proteus on a machine for the first
time and run them:
```shell
cd build
ctest ./
```

We are keen to help with bugs or any other issues in our repo's
[issues](https://github.com/Olympus-HPC/Mneme/issues) page!
