# Limitations

Mneme provides a practical record–replay and autotuning workflow for GPU kernels, but there are currently a few known limitations. 
These are not fundamental design blockers, but they may affect certain applications today.

---

## 1. No Support for Managed Memory

Mneme does **not** currently support CUDA Unified / Managed Memory (`cudaMallocManaged`).

**Implications:**
- Kernels that rely on managed memory allocations may fail during replay.
- Memory state reconstruction assumes explicit device memory allocations (`cudaMalloc`, `hipMalloc`).

**Workaround:**
- Replace managed memory with explicit host–device memory transfers.
- Use pinned host memory and explicit `cudaMemcpy` where possible.

---

## 2. Restrictions on Global Variables

Global variables are supported **as long as their addresses are not captured by another global variable**.

**Unsupported pattern:**
```cpp
__device__ int g_value;
__device__ int* g_ptr = &g_value;  // ❌ not supported
```

Supported pattern:
```cpp
__device__ int g_value;             // ✅ supported
```

Why this matters:

- Mneme records and reconstructs global memory symbols independently.
- Address aliasing between globals complicates relocation and replay correctness.

Workaround:

- Avoid global pointer aliasing.
- Initialize pointer relationships inside a kernel or host-side setup code instead.

## 3. CUDA RDC (Relocatable Device Code) Is Untested

CUDA Relocatable Device Code (RDC) is not tested and may not work reliably.

Implications:

- Multi-translation-unit device code, device-side linking, and dynamic device symbol resolution may fail.
- Kernel replay may break when kernels depend on symbols defined in separate device objects.

Current status:
- RDC-related issues have not yet been systematically evaluated.

If you need this:

- Please open a GitHub issue with a minimal reproducer.
- RDC support is planned but not yet prioritized.

## Reporting Issues or Requesting Support

If any of these limitations block your use case, please:


Open a GitHub issue
Include:

- CUDA / HIP version
- LLVM version
- Mneme version
- Minimal reproducer
- Expected vs. actual behavior

Your feedback directly drives prioritization.
