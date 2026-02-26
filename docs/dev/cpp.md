# C++ Development

- **C++ API reference (Doxygen):** [Open the generated API docs](./html/index.html)

If the link 404s locally, ensure Doxygen output is generated into `docs/dev/index/`
before running `mkdocs build` or `mkdocs serve`.

---

## Annotation API (`mneme/MnemeAnnotation.hpp`)

Mneme ships a lightweight C++ header that lets application code attach
verification metadata to device pointers.  The metadata is recorded
alongside the device memory snapshot and consumed during replay to
perform tolerance-aware comparison.

### Include and link

```cpp
#include "mneme/MnemeAnnotation.hpp"
```

The `add_mneme(target)` CMake function (provided by
`MnemeFunctions.cmake`) links the annotation runtime library
(`mnemert`) automatically — no extra `target_link_libraries` call is
needed.

### Quick example

```cpp
double *d_buf = nullptr;
cudaMalloc(&d_buf, N * sizeof(double));

// Typed overload: deduces BuiltinDType from the pointer type.
mneme::annotate<double>(d_buf, mneme::Metadata{
    .threshold      = 1e-6,
    .threshold_kind = mneme::ThresholdKind::Relative,
    .norm           = mneme::Norm::Linf,
    .tag            = std::string("my_buffer"),
});
```

See **[Usage → Verification](../usage/verification.md)** for the full
field reference and end-to-end examples.

