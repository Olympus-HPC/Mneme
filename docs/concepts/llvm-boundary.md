# LLVM IR Boundary

Mneme treats **LLVM IR as the boundary** between recording and replay.

During recording, Mneme captures and stores the kernel code in LLVM IR form.
During replay, Mneme recompiles that LLVM IR (optionally applying transformations)
to execute the kernel in isolation. This design makes replay **compiler-driven**,
transparent, and tunable.

---

## Why LLVM IR?

Mneme needs a stable handle to the kernel code that is:

- precise enough to recompile and optimize,
- structured enough to support transformations and specialization, and
- portable across tooling (LLVM passes, disassemblers, analyzers).

Source code is a poor boundary for this purpose because it is fragmented
across translation units and heavily shaped by language semantics
(templates, macros, headers, and ABI details), and there is non-negligible time spent lowering
this information into an intermediate representation. Conversely, recording a
final device binary is also limiting: much of the high-level structure is
lost, and many compiler-level transformations become difficult or impossible.

LLVM IR sits at a useful middle ground:

- the type system is explicit and normalized,
- module boundaries are well defined,
- kernels and their dependencies are available in a consistent representation,
- standard LLVM tools can inspect and transform it.
- Lowering LLVM IR to device requires minimal external parameters, thus we can use Proteus JIT capability to efficiently build executables during replay. 

---

## What Mneme captures at the IR boundary

Mneme does **not** record the entire application in LLVM IR form.
Instead, Mneme captures the LLVM IR required to replay a specific kernel.

Conceptually, this includes:

- the kernel entry point, and
- the **transitive dependency closure** needed to compile it in isolation
  (device functions, helpers, and any IR-level dependencies required by codegen).

This captured IR is stored as recorded LLVM bitcode (`.bc`) and is referenced
by the recording database.

See **[Execution Phases](mneme-phases.md)** for how this IR is discovered and embedded at build time,
and **[Artifacts](artifacts.md)** for how it is stored and referenced.

---

## What happens at replay time

At replay, Mneme uses the recorded LLVM IR as input to a compilation and execution flow:

1. Load the recorded `.bc` module(s)
2. Select the kernel entry point
3. Optionally apply transformations, specializations and optimizations (passes / specialization)
4. Lower to a device object (codegen)
5. Load the resulting code and execute the kernel with the recorded (or overridden)
   replay configuration

This is why replay configuration includes compilation knobs such as:

- pass pipelines (e.g., `default<O3>`)
- optimization levels for backend code generation
- specialization options (e.g., specializing dims or arguments)

See **[Replay Configuration](replay-configuration.md)** for the full set of replay-time knobs.

---

## Pipelines and transformations

Replay supports choosing a compilation **pipeline** (also referred to as
a pass pipeline). This pipeline controls which LLVM passes are applied
before code generation.

From the user’s perspective, this is the `passes` argument to `mneme replay`:

```bash
mneme replay -rdb <record.json> -rid <dynamic-hash> "default<O3>"
```

The `passes` argument is a string representation of a pipeline as described by the 
LLVM tool `opt`. Internally, for efficiency, Mneme uses the LLVM APIs to apply these transformations
instead of invoking sub-processes and communicating through the file system.

Mneme uses LLVM IR so users can also:

- experiment with different optimization pipelines,
- debug performance regressions at the IR level, and
- explore specialization strategies without touching the original application.
- tune kernel launch configurations

## Inspecting and modifying recorded IR

Recorded LLVM IR artifacts are standard LLVM bitcode files (.bc) and can be
inspected using common LLVM tools.

### Typical workflows include:

Disassemble to text IR:
```bash
llvm-dis RecordedIR_<static-hash>.bc -o -
```

### Apply an LLVM opt pipeline:

```bash
opt -O3 RecordedIR_<static-hash>.bc -o optimized.bc
```

### Capture the replay-time IR for debugging:

```bash
mneme replay ... --output-ll out.ll "default<O3>"
```

See [Artifacts](artifacts.md) for where recorded .bc files are stored and how they are referenced.

## Practical implications and limitations

LLVM IR is a powerful boundary, but it comes with practical constraints:

- Toolchain coupling: Recorded IR must remain compatible with the LLVM toolchain used
for replay. Mneme is tested with ROCm-provided LLVM versions (e.g., LLVM 18/19/20).
- Not full-program replay: Mneme records kernel-level IR, not host-side control flow.
- Backend differences matter: Code generation decisions (and performance) may vary across
GPU targets, LLVM versions, and codegen options.
- Undefined behavior remains undefined: If the kernel or its inputs rely on UB, replay may
not reproduce results reliably.
- Advanced workflows: Power users may modify application source code,
  rebuild the binary, and extract or replace the embedded LLVM IR
  sections using external tooling. Replay correctness depends on the
  transformed code preserving compatible device memory layouts.

These constraints are a direct consequence of making replay a compilation-driven process.

## Why this boundary matters for Mneme

Choosing LLVM IR as the boundary enables Mneme to:

- decouple recording from replay and tuning,
- treat kernel replay as a first-class compilation workflow,
- integrate cleanly with LLVM tooling and custom passes,
- support systematic tuning over compiler and launch parameters.

In short: LLVM IR makes Mneme a replay + compiler experimentation platform, not just
a tracing or profiling tool.
