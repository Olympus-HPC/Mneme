# Examples

The examples directory contains benchmarks derived from the [HecBench](https://github.com/zjin-lcf/HecBench) suite, adapted to demonstrate Mneme's **kernel-level record, replay, and tuning** capabilities.

Mneme enables users to **isolate** GPU kernels from larger applications by capturing the exact code (LLVM IR) and device memory state required for execution. This isolation allows for rapid experimentation, search space exploration, and auto-tuning without the overhead of running the full application or managing complex host-side dependencies.

The examples provided here serve as a reference for:

*   **Integration**: How to link Mneme with existing C/C++ build systems (CMake).
*   **Recording**: Capturing kernel executions at runtime.
*   **Tuning**: Using Python-based tools like **Optuna** to explore replay configurations (block size, grid size, etc.) and optimize performance.

For instance, the `miniFE` example explicitly demonstrates how to define a search space over kernel parameters and drive the replay engine to find optimal configurations.

## Building

All examples are built using CMake. For each benchmark, the build system generates three distinct binaries:

1.  **Original**: The baseline application without any instrumentation.
2.  **Proteus**: The application instrumented with the base Proteus runtime.
3.  **Mneme**: The application fully instrumented with Mneme, enabled for **recording** and **replay**.

### Requirements

To build the examples, you need to provide the installation paths for **Mneme**, **Proteus**, and your **LLVM** installation (if not in standard paths).

### Build Command

You can build the examples by configuring with CMake. You must enable HIP backend support using the`WITH_MNEME_EXAMPLE_HIP` flags.

```bash
mkdir build && cd build
cmake -DCMAKE_C_COMPILER=$(mneme config cc) \
      -DCMAKE_CXX_COMPILER=$(mneme config cxx) \
      -DCMAKE_PREFIX_PATH=$(mneme config cmakedir)
      -DWITH_MNEME_EXAMPLE_HIP=On \
      ../examples/hecbench
make -j
```

## WSM5

**Location:** `examples/hecbench/wsm5`

WSM5 (WRF Single Moment 5-class Microphysics) is a kernel extracted from the Weather Research and Forecasting (WRF) model. It simulates microphysics processes including vapor, rain, snow, cloud ice, and cloud water.

The implementation relies on CUDA kernels to perform the heavy lifting. The `main.cpp` orchestrates the data movement and kernel launches.

- **Key Files**:
    - `main.cpp`: Setup and execution of the WSM5 kernel.
    - `kernel.h`: formatting and logic for the CUDA kernel.
    - `tune.py`: Tuning script exploring kernel launch parameters.
    - `tune_passes.py`: Tuning script exploring compiler optimizations.

### Recording

To record the execution of the WSM5 kernel, you use the `mneme record` command. The application takes a single argument indicating the number of repetitions.

```bash
# syntax: mneme record -rdb <database_dir> -- <application> <args>
mneme record -rdb wsm5_db -- ./build/examples/hecbench/wsm5/wsm5-mneme 1
```

This will create a `wsm5_db` directory containing the recording database and artifacts (memory snapshots, LLVM IR).

### Tuning

Once recorded, you can tune the kernel using the provided Python scripts. These scripts demonstrate how to use Mneme's Python API to drive replay with different configurations.

**1. Kernel Parameter Tuning (`tune.py`)**

This script focuses on tuning **kernel launch parameters** and **specialization**. It explores a search space defined by:

- Block dimensions (`block_dim_x`, `block_dim_y`, `block_dim_z`)
- Grid dimensions (`grid_dim_x`, `grid_dim_y`, `grid_dim_z`)
- Specialization of kernel arguments
- Launch bounds

```bash
# syntax: ./tune.py --record-db <database_dir> --record-id <kernel_id>
./examples/hecbench/wsm5/tune.py --record-db wsm5_db/123456789.json --record-id 987654321
```

**2. Compiler Pass Tuning (`tune_passes.py`)**

This script focuses on tuning the **code generation** process itself. Instead of just changing launch parameters, it modifies how the kernel is compiled from the recorded LLVM IR. It explores:

- Optimization pipelines (e.g., `-O1`, `-O2`, `-O3`, custom pass lists)
- Backend code generation options
- Specialization (as it interacts with optimization)

```bash
./examples/hecbench/wsm5/tune_passes.py --record-db wsm5_db/123456789.json --record-id 987654321
```

**Conceptual Difference**:
*   `tune.py` optimizes **how the kernel is run** (threads per block, grid size).
*   `tune_passes.py` optimizes **how the kernel is built** (compiler optimizations, register allocation strategies).

Both approaches rely on Mneme's unique ability to **recompile** the recorded LLVM IR on-the-fly during replay.

## Bezier Surface

**Location:** `examples/hecbench/bezier-surface`

This example computes a Bezier surface. It demonstrates a basic integration where the application can run on GPU and contrasts the computer output with a CPU implementation.

- **Key Files**:
    - `main.cpp`: Contains the host and device code for Bezier surface calculation.
    - `tune.py`: Python script for auto-tuning the kernel parameters and compiler passes.

### Recording

To record the execution of the Bezier Surface kernel, you use the `mneme record` command. The application requires an input file and an output size.

```bash
# syntax: mneme record -rdb <database_dir> -- <application> -f <input_file> -n <size>
mneme record -rdb bezier_db -- ./build/examples/hecbench/bezier-surface/bezier-mneme -f examples/hecbench/bezier-surface/input/control.txt -n 8192
```

### Tuning

The provided `tune.py` script for this example combines both parameter tuning and compiler pass selection into a single search space. It uses `PipelineManager` to generate a list of potential optimization pipelines.

```bash
# syntax: ./tune.py --record-db <database_dir> --record-id <kernel_id>
./examples/hecbench/bezier-surface/tune.py --record-db bezier_db/123456789.json --record-id 987654321
```

This script exhaustively searches through combinations of:

- **Specialization**: Toggling kernel argument specialization.
- **Launch Bounds**: Enabling/disabling launch bounds.
- **Optimization Pipelines**: Iterating through standard levels (`-O1`, `-O2`, `-O3`) and a procedurally generated set of custom pass sequences.

## MiniFE

**Location:** `examples/hecbench/miniFE`

MiniFE is a Finite Element mini-application which implements a couple of kernels representative of implicit finite-element applications.

This example is particularly notable for its integration with **Optuna** for auto-tuning. The `tune.py` script provided in this directory demonstrates a workflow where:

1. A kernel execution is recorded using Mneme.
2. An `EntireSpace` class defines a search space for tuning parameters like `warp_fraction`, `grid_fraction`, and `max_threads`.
3. An `AsyncReplayExecutor` is used to replay the recorded kernel with different configurations proposed by Optuna to find the optimal speedup.

- **Key Files**:
    - `tune.py`: An advanced example of defining a search space and using Optuna to tune a recorded kernel execution.
    - `src/`: Source code for the miniFE application.

### Recording

To record the execution of the MiniFE kernel, you use the `mneme record` command. The application requires dimensions for the problem size (x, y, z).

```bash
# syntax: mneme record -rdb <database_dir> -- <application> -nx <nx> -ny <ny> -nz <nz>
mneme record -rdb miniFE_db -- ./build/examples/hecbench/miniFE/miniFE-mneme -nx 220 -ny 200 -nz 190
```

### Tuning

The `tune.py` script for MiniFE uses the **Optuna** library to drive the search for optimal kernel parameters. Unlike the other examples that use a simple exhaustive search, this demonstrates how to integrate Mneme with external optimization frameworks.

```bash
# syntax: ./tune.py --record-db <database_dir> --record-id <kernel_id>
./examples/hecbench/miniFE/tune.py --record-db miniFE_db/123456789.json --record-id 987654321
```

Keys parameters tuned in this example include:

- **Warp Fraction**: Adjusting the number of active warps.
- **Grid Fraction**: Scaling the grid size.
- **Max Threads**: Limiting the maximum threads per block.

