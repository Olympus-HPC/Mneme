# Installation and Usage 
<!-- Currently, Mneme is distributed using its git repo to build and install it.  -->
Mneme is open-source and available on GitHub. We recommend using the **latest develop** branch version, which is well tested and
robust while including the most recent features.

Mneme builds and installs the following key components: 
- A LLVM plugin pass, `libregdeviceir.so`, which is used in the first phase to generate a _recordable_ executable.
- A recording library, `librecord.so`, which is used in the second phase to facilitate the actual recording of device and host memory data from the recordable executable generated in phase 1.
- A user-friendly command-line tool, `mneme execute|tune`, that allows users to execute, optimize, and tune the recorded traces. This tool utilizes the `libmneme.so` library. 

Further, Mneme also build the following two libraries that are utilized internally: 
- A profiling library, `libmneme_profile.so` , for collection of peformance profiles on for HIP kernels (ROCm profiling), and,  
- A stub library, `libmneme_shallow.so` , that enables building of binaries with and without recording, reducing user's build effort
- A command-line tool, `replay`, that allows users to replay the recorded traces. 

A user must integrate the LLVM pass in their build system and utilize the recording library to generate the traces. They can then leverage the `mneme` command-line tool for replay and tuning. 
We provide information on how to integrate Mneme with your application in the
[Integration](integration.md) section, along with an example of replay and tuning.


## Quick Start
```
# HIP installation set up 
module load rocm/6.3.1
export LLVM_INSTALL_DIR=${ROCM_PATH}

# Clone and install
git clone https://github.com/Olympus-HPC/Mneme.git
pip3 install ./
```

The above steps install the `mneme` package in the relevant install directory. It also creates a `build` folder and a `third_party` folder. The `build` folder contains the libraries mention in the previous subsection. 

## Building
The `Quick Start` section provides instructions for a user-friendly build with `pip` on AMD GPUs. Advanced users can utilize the `cmake` build instead of the `pip`build if desired, which is detailed below. 

Please note that currently, NVIDIA GPUs (CUDA builds) are only supported through `cmake`. 

### Dependencies
Mneme depends on LLVM (ROCm@6.2 or ROCm@6.3), [spdlog](https://github.com/gabime/spdlog.git), and [Proteus](https://github.com/Olympus-HPC/proteus/). 

Internally, the project uses `cmake` for building and depends on an LLVM installation.  The top-level
`CMakeLists.txt` has the following (binary) build options:

* `MNEME_ENABLE_TESTS`: builds Mneme tests.
* `MNEME_ENABLE_HIP`: enables HIP support.
* `MNEME_ENABLE_CUDA`: enable CUDA support (**support for cuda is work in progress**).
* `MNEME_ENABLE_DEBUG`: logs debugging information (for developers).
* `MNEME_ENABLE_LOGGER`: Enalbes Mneme logging.

The script clones `proteus` and `spdlog`, builds and installs them,  and then installs Mneme.

Please check the build process of our CI [here](scripts/gitlab/ci-build-test.sh). 

### CMake Build Steps (CUDA example)

TBD.

### Testing
If using a `cmake` setup, it is advised to enable tests when deploying Mneme on a machine for the first
time and run them:
```shell
cd build
ctest ./
```

We are keen to help with bugs or any other issues in our repo's
[issues](https://github.com/Olympus-HPC/Mneme/issues) page!