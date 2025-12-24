# Static vs Dynamic Hashes

Mneme distinguishes between static kernel identity and dynamic
kernel execution identity. This distinction is fundamental to how
Mneme records, organizes, and replays GPU kernels.

Understanding static and dynamic hashes helps explain:
- why recording databases are structured the way they are,
- how multiple kernel launches map to a single kernel definition, and
- how replay and tuning operate on individual executions.

## Static kernel identity (static hash)

A static hash uniquely identifies a GPU kernel by its code.
Conceptually, the static hash represents:

- a single kernel definition in the application. Static hashes are computed after the resolution of C++ templates. As such, it takes into account template parameters and kernels defined with different template types will result in different static hashes,
- independent of how many times it is launched or with which parameters.

### Properties

- Stable across executions of the same binary,
- Independent of runtime launch configuration,
- Represents the kernel’s source-level identity,
- Used as the primary identifier for recorded kernels

In practice, the static hash is derived from the kernel’s LLVM IR and its
transitive dependencies, as captured during the instrumentation phase.

### Where it appears

The static hash:

- names the recording database file (<static-hash>.json), 
- names the recording prologue and epilogue snapshots file (<static-hash>.json, DeviceState.epilogue.<static-hash>.<dynamic-hash>.mneme), 
- identifies the kernel in recording and replay workflows, and
- links recorded LLVM IR to all runtime executions of that kernel.


## Dynamic execution identity (dynamic hash)

A dynamic hash identifies a specific runtime execution of a
kernel.

Each time a kernel is launched, its execution context may differ due to:

- grid dimensions,
- block dimensions,
- shared memory size.

Mneme assigns a dynamic hash to each distinct execution context observed
during recording.

!!! note
    Future versions of Mneme may refine dynamic hash computation further
    by incorporating additional runtime parameters, such as scalar kernel
    arguments.

#### Properties

- Specific to a single kernel launch configuration,
- Unique within the scope of a static kernel,
- Represents how the kernel was executed, not what kernel it is.

During recording, Mneme uses both the static hash and dynamic hash as a
**hierarchical cache key** to determine whether a given kernel execution
has already been recorded. If a matching entry exists, recording of that
execution is skipped.

## Relationship between static and dynamic hashes

The relationship can be summarized as:

> One static hash → many dynamic hashes

The static hash identifies the kernel code. Each dynamic hash corresponds to one execution instance of that
kernel with a particular runtime configuration.

This relationship is reflected directly in the recording database
```bash
static kernel (static hash)
│
├── instance A (dynamic hash)
├── instance B (dynamic hash)
└── instance C (dynamic hash)
```

Each instance may have:
- different grid/block dimensions,
- different memory usage patterns,
- different performance characteristics.

## Why this distinction matters

Separating static and dynamic identity enables Mneme to:

- **avoid code duplication:** LLVM IR is stored once per static kernel, even if the kernel is executed many times,
- **record realistic inputs:** Multiple dynamic instances capture real application behavior across different execution contexts,
- **target replay and tuning precisely:** Replay operates on a specific dynamic hash, allowing fine-grained experimentation without ambiguity,
- **scale recording efficiently:** Large applications often launch the same kernel thousands of times; static/dynamic separation keeps recordings manageable.

## Interaction with replay and tuning

During replay:

- the user selects a static kernel (via the recording database), and
- chooses a dynamic instance (via the -rid / record-id option).

Replay configurations may:

- reproduce the recorded dynamic configuration exactly, or
- override parts of it (e.g., block size, specialization).

Tuning workflows iterate over replay configurations starting from a
dynamic hash baseline, ensuring correctness is always validated
against a real recorded execution.

## Summary

- **Static hash** identifies which kernel is being considered,
- **Dynamic hash** identifies how that kernel was executed,
- Mneme records one static kernel with many dynamic execution instances,
- Replay and tuning operate at the dynamic-hash level while sharing
static kernel code.

This separation is central to Mneme’s scalability, correctness, and
flexibility.
