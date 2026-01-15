# Installation

Mneme is open-source and available on GitHub. We recommend using the
**latest `develop` branch**, which is well tested and includes the most
recent features.

---

## Dependencies and compatibility

Mneme depends on a small set of external components. Compatibility is
defined in terms of supported ROCm versions, Python versions, and a
specific Proteus commit.

### ROCm and Python compatibility matrix

The following table summarizes the configurations that are regularly tested
and known to work with Mneme. These combinations are validated in CI and/or
on internal test systems.

| ROCm version | Python 3.9 | Python 3.10 | Python 3.11 | Python 3.12 |
|-------------|------------|-------------|-------------|-------------|
| **6.3**     | ✅         | ✅          | ✅          | ✅          |
| **6.4**     | ✅         | ✅          | ✅          | ✅          | 
| **7.0**     | ⏳         | ⏳          | ⏳          | ⏳          | 

#### Notes

- ✅ **Supported**: configuration is tested and fully supported.
- ⚠️ **Experimental**: expected to work, but not yet covered by continuous
  integration or full test coverage.
- ⏳ **Coming soon**: planned support; not yet available.
- Mneme relies on the LLVM/Clang toolchain shipped with the corresponding
  ROCm release.
- Python support refers to the Python version used to run the Mneme CLI and
  Python API; it does not affect device compilation.

> Support for additional ROCm versions and CUDA backends is planned.

### Proteus dependency and compatibility

Mneme depends on the Proteus JIT and LLVM transformation infrastructure.
At present, neither Mneme nor Proteus follows a formal versioning scheme.

Compatibility between Mneme and Proteus is therefore defined at the
**commit (SHA) level**.

Mneme is regularly tested against a specific Proteus commit known to be
compatible. Users building Mneme from source are strongly encouraged to
use the corresponding Proteus commit to avoid incompatibilities.

!!! note
    Mneme requires Proteus to be built and installed as a **shared library**.
    Mneme intercepts selected Proteus entry points to inject custom
    record and replay functionality, which is not possible with a
    static-only Proteus build.

#### Tested Proteus commit

- Repository: https://github.com/Olympus-HPC/Proteus
- Commit: `1d21c00008061704459a9b20300556e962c89043`  
- Tested with: Mneme `develop`

!!! note
    Mneme may not be compatible with the latest Proteus `main` branch at all times.
    Proteus is under active development, and changes to core components may
    temporarily break compatibility with Mneme.

    Users are strongly encouraged to use the tested Proteus commit listed above.

### spdlog dependency and compatibility

Mneme depends on [spdlog](https://github.com/gabime/spdlog) for emitting logging
messages during recording and when using the C++ bindings.

Compatibility between Mneme and spdlog is defined through **library versioning**.
Mneme is currently tested against and pins spdlog version **1.15.0**.

### LLVM and toolchain requirements

Mneme requires a working **Clang/LLVM toolchain** for IR instrumentation,
kernel replay, and specialization.

The following tools and libraries must be available:

- `clang` / `clang++`
- LLVM libraries

These tools are provided by the **LLVM distribution bundled with ROCm**.
Mneme is currently tested with **LLVM 18 and LLVM 19** as shipped by
supported ROCm releases.

!!! note
    Mneme expects the ROCm-provided LLVM toolchain to be used.
    System-installed LLVM versions may not be compatible.
    Users are encouraged to rely on `mneme config cc`, `mneme config cxx`,
    and `mneme config cmakedir` when building applications with Mneme.


## Installation steps

### User installation (recommended)

This installation method is recommended for users who want to use Mneme
to record and replay kernels.

```bash
git clone https://github.com/Olympus-HPC/Mneme.git
cd Mneme
export LLVM_INSTALL_DIR=${ROCM_PATH}
pip install .
```

This installs the Mneme CLI (mneme) and Python bindings along with all
tested runtime dependencies.

### Developer installation

This installation method is recommended for contributors and developers
working on Mneme itself.

```bash
pip install -e .
```

Editable mode installs Mneme in-place, allowing local source changes to be
picked up without reinstallation.

### Optional: Using an external Proteus installation

Mneme can be configured to use an **existing Proteus source tree or a
pre-built Proteus installation** via environment variables.

This is intended for advanced users and application developers who
already build or ship binaries linked against Proteus. In such cases,
applications do **not** need to be rebuilt specifically for Mneme:
recording and replay functionality can be enabled by preloading Mneme
at runtime.

This capability is not part of the default installation path and is
currently less extensively tested than the bundled Proteus workflow.

### Optional: Using an external Proteus installation

Mneme can be configured to use an **existing Proteus source tree** or a
**pre-built Proteus installation** via environment variables.

This option is intended for advanced users and application developers who
already build or ship binaries linked against Proteus. In such cases,
applications do **not** need to be rebuilt specifically for Mneme:
recording and replay functionality can be enabled by preloading Mneme
at runtime.

This capability is not part of the default installation path and is
currently less extensively tested than the bundled Proteus workflow.

#### Environment variables

The following environment variables may be used to point Mneme to an
external Proteus installation:

- `PROTEUS_SRC`: Path to a Proteus source tree. When set, the Mneme
  installer will configure and build Proteus from this source.
- `PROTEUS_DIR`: Path to an existing Proteus installation prefix.
  This directory must allow `find_package(Proteus)` to succeed.

When either of these variables is set, Mneme will use the specified
Proteus installation instead of the internally managed one.

When both of these variables are set, `PROTEUS_DIR` takes priority.

!!! note
    This workflow is intended for advanced use cases.
    Users are encouraged to start with the default installation unless
    they already have an existing Proteus-based application.

### Verifying the installation

Mneme uses pytest for its Python test suite.

First, install the test dependencies:

```bash
pip install pytest
```

Then run the tests:

```
pytest python/tests
```

Successful completion of the test suite indicates that Mneme and its
Python bindings are correctly installed.

## Next steps

Once Mneme is installed and the test suite completes successfully,
proceed to **Getting Started** for a guided, end-to-end example of
building, recording, and replaying a GPU kernel with Mneme.

