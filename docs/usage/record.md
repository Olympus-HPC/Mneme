# Record

The `mneme record` command executes an application while transparently
capturing GPU kernel executions and the associated device memory state.
For full CLI details, see [CLI → mneme record](cli.md#mneme-record).

Recording does not require modifying application source code.  However,
if you want **tolerance-aware verification** during replay, you can
optionally annotate device pointers with comparison metadata before the
kernel launches that will be recorded.

## Choosing what to capture

By default a recorded prologue and epilogue contain every live device
allocation in the process, whether or not the kernel can reach it.
For applications with a large resident heap this makes snapshots large
and makes every replay iteration re-upload memory the kernel never
touches.

`--capture-mode reachable` restricts the snapshot to the allocations the
launch can actually address:

- every device global registered for the kernel's binary (unchanged from
  the default), and
- every tracked allocation that a pointer-typed kernel argument points
  into, including pointers stored inside `struct`, lambda, or functor
  arguments passed by value.

```bash
mneme record --capture-mode reachable -- <application> [args...]
```

Pointer slots are derived from the kernel's LLVM IR, so no annotation is
needed. Replay still reserves the full recorded virtual address range, so
recorded pointer values remain valid; allocations that were not captured
are simply left uninitialized. The recording JSON reports the mode in its
`CaptureMode` field.

Reachable capture has limits that the default mode does not:

- An argument declared as an integer (for example `uintptr_t`) is not a
  pointer slot even if it holds a device address, so its buffer is not
  captured.
- Only pointers held directly in the arguments are followed. A buffer that
  is reachable solely through a pointer stored in device memory (such as
  a device-resident struct of pointers) is not captured.
- A pointer whose target is neither a tracked allocation nor a registered
  global, such as managed memory, prints a `[mneme]` warning on stderr and
  is left out. Such a kernel cannot be replayed correctly in either mode.

## Annotating buffers for verification

By default, replay verification uses exact byte-wise comparison between
the replayed and recorded epilogue snapshots.  For kernels that produce
floating-point results, this is often too strict.

Mneme provides a lightweight C++ API — `mneme::annotate()` — that lets
you attach a data type, error threshold, aggregation norm, and an
optional tag to any device pointer.  The metadata is captured inside
the recorded snapshot and used automatically during every subsequent
replay.

```cpp
#include "mneme/MnemeAnnotation.hpp"

mneme::annotate(d_output, mneme::Metadata{
    .builtin        = mneme::BuiltinDType::F64,
    .threshold      = 1e-6,
    .threshold_kind = mneme::ThresholdKind::Relative,
    .norm           = mneme::Norm::Linf,
    .tag            = std::string("output_vector"),
});
```

Annotations must be applied **before** the kernel launch they should
affect.  You can update the annotation on the same pointer between
launches to record different tolerance policies for different dynamic
instances of the same kernel.

For the full API reference, supported data types, threshold semantics,
and a complete example, see **[Usage → Verification](verification.md)**.
