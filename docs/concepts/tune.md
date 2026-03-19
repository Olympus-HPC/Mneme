# Tuning Model

Mneme’s tuning model builds on replay by turning kernel execution into an
**optimization problem**: find the replay configuration (or set of configurations)
that minimizes a user-defined objective, typically **execution time**, while
preserving **correctness**.

The key idea is to separate:

- **what you want to tune** (your tunable parameters), from
- **what Mneme can execute** (a concrete replay configuration).

This separation is what makes Mneme’s tuning flexible and extensible.

---

## Mental model

Tuning in Mneme is the loop:

1. Start from a **recorded kernel instance** (a dynamic hash baseline).
2. Define a **search space** (your knobs, constraints, and derived parameters).
3. For each sampled point:
   - map parameters → **replay configuration**
   - replay & measure
   - validate correctness
4. Keep the best-performing configurations and record results for later reuse.

---

## Core abstractions

### Recorded execution baseline

Tuning always starts from recorded data:

- a recording database (`--record-db`)
- a selected kernel instance (`--record-id` / `-rid`)

This provides:

- a stable kernel identity,
- a “known-good” launch configuration,
- recorded device memory state for correctness checking, and
- recorded LLVM IR as the transformation surface.

### Replay configuration space

A **replay configuration** is what Mneme can actually execute.

It controls:

- launch dimensions (`grid`, `block`, `shared_mem`)
- specialization toggles (`specialize`, `specialize_dims`)
- compilation/codegen settings (`passes`, `codegen_opt`, `codegen_method`)
- optional launch-bounds knobs (`set_launch_bounds`, `max_threads`, `min_blocks_per_sm`)
- replay hygiene (`prune`, `internalize`)

See **[Replay Configuration](replay-configuration.md)** for the full conceptual view.

### Search space

A **search space** defines the user-facing tuning problem.

It is intentionally **orthogonal** to replay configuration space: the user can
tune *any parameters they care about*, not necessarily a 1:1 match with Mneme’s
internal knobs. However, the user needs to **derive** from the sampled search space a 
valid mneme replay configuration.

In the Python API, this is expressed as a `SearchSpace` object with:

- `dimensions()`  
  Declares tunable knobs (integer ranges, categorical choices, booleans, pipelines, reals, …).
- `constraints(params)`  
  Filters invalid or undesired samples (device limits, domain rules, heuristics).
- `derived(params) -> ExperimentConfiguration`  
  **Projects** the user parameter vector into a concrete replay configuration.
- `baseline()`  
  A reference configuration (typically “use the recorded dims + recorded shared memory”).

---

## Why “search space ≠ replay configuration” matters

Many tuning problems require a different parameterization than “raw replay knobs”.

Examples:

- You want to tune a **fractional** knob (e.g., 0.0–1.0) and map it to a **discrete**
  device-legal value (e.g., `max_threads` must be a multiple of 64 and ≥ threads/block).
- You want to tune only *one* dimension (`block_dim_x`) and keep everything else fixed.
- You want to express derived relationships:
  - `grid.x = f(problem_size, block.x)`
  - `max_threads = g(block.x, fraction)`
  - “if `set_launch_bounds` is false, ignore launch-bound parameters”

Mneme’s tuning model supports these patterns directly via `derived()` and `constraints()`.
Overall, constraints can identify non-valid configurations early on and avoid replaying them. 
However, usually it is more efficient to always derive valid configurations as the generation
of a sample from the search space can take time.

---

## Example: projecting a continuous knob to a discrete one

Assume that the user tunes:

- `max_threads_fraction ∈ [0.0, 1.0]` (continuous)

But Mneme needs:

- `max_threads ∈ {block_threads, block_threads+64, …, 1024}` (discrete and constrained)

A `derived()` method implements the projection:

- sample `max_threads_fraction`
- compute `proper_range`:
  - start at `block_dim_x`
  - move toward 1024 in steps of 64
- emit a valid `ExperimentConfiguration`

This is a canonical Mneme tuning pattern: **express the problem naturally**, then
map it to **valid replay configurations**.

---

## Replay of derived configurations

Once a replay configuration has been derived from a search-space sample,
Mneme executes it through an **executor abstraction**.

The executor is responsible for:

- submitting replay configurations for execution,
- managing concurrency and resource utilization, and
- collecting and returning execution results.

This abstraction decouples *tuning logic* from *execution mechanics*.

---

### Executor model

Mneme provides executors that can evaluate replay configurations
**asynchronously** and **in parallel**, enabling efficient use of
available system resources, including multiple GPUs.

At a high level:

- the tuning loop submits replay configurations to an executor,
- the executor schedules and executes them, and
- results are returned to the tuning logic as they complete.

This allows tuning frameworks (Optuna, exhaustive search, custom drivers)
to remain simple and synchronous from the user’s perspective while
execution proceeds asynchronously under the hood.

---

### Asynchronous execution and workers

Internally, the executor manages a pool of **replay workers**:

- each worker is responsible for executing one replay at a time,
- workers may be bound to specific devices (e.g., GPU IDs), and
- multiple workers can execute concurrently.

Each worker is monitored by a dedicated **control thread**. This thread:

- launches the replay process,
- monitors its execution status,
- captures results or failures, and
- performs cleanup when execution completes or fails.

This separation between the **main executor** and **worker-monitor threads**
is intentional and enables robust, asynchronous operation.

---

### Resiliency and fault isolation

Replay and tuning workflows may encounter failures, for example due to:

- invalid or overly aggressive code transformations,
- illegal launch configurations,
- code generation or compilation errors, or
- runtime device failures.

Mneme’s executor design isolates such failures:

- a failed replay does **not** crash the tuning process,
- the worker handling that configuration is safely terminated,
- the executor reports failure status to the caller, and
- subsequent configurations can continue executing.

This makes it practical to explore large or risky search spaces where some
configurations are expected to fail.

---

### Scaling across devices

Executors can be configured to:

- use multiple GPUs,
- control the number of concurrent workers, and

This allows Mneme to scale from:
- single-GPU exploratory tuning, to
- multi-GPU, high-throughput tuning runs.

## Sampling strategies and optimizers

Mneme separates *what to sample* from *how to sample*.

- `SearchSpace` defines the knobs and mapping.
- A sampling strategy controls how points are explored.


This enables:

- exhaustive search for small spaces,
- randomized exploration,
- Bayesian/TPE optimization,
- multi-objective optimization (NSGAIIS samplers for advanced workflows).

Mneme instead of implementing these samping/tuning technologies relies on the
state of the art `optuna` library to traverse and sample the search space.

---

## Evaluation and correctness

Each sampled configuration is executed via an executor:

- `AsyncReplayExecutor.evaluate(config) -> ExperimentResult`

A key feature is that evaluation includes **correctness validation**:

- `result.verified` indicates whether replay matched the expected recorded behavior.

Typical tuning logic is:

- if invalid config: assign worst score and continue
- if replay fails or not verified: penalize and continue
- otherwise: measure performance and report the objective value

This creates a robust “correctness-first” tuning loop.

---

## Metrics and objective functions

Most users start with:

- objective: minimize mean execution time  
  `avg_time = mean(result.exec_time)`

But Mneme does not enforce the objective. You can optimize:

- median time,
- percentile time,
- codegen time vs exec time tradeoffs,
- object size,
- correctness + performance multi-objective,
- hardware counters (when integrated).

---

## Persistence and reproducibility

A tuning run is only useful if results can be reproduced.

Mneme supports reproducibility via:

- configuration hashing: `config.hash()`
- serializable configs and results:
  - `config.to_dict()`
  - `result.to_dict()`

---

## Summary

Mneme tuning is defined by three separable layers:

- **Replay configuration**: what Mneme executes
- **Search space**: how you define your tuning problem
- **Sampling/optimizer**: how you explore the space

The critical abstraction is `derived(params)`: it maps your problem’s parameterization
to a valid replay configuration, giving you full freedom to express constraints,
derived relationships, and grounded discrete choices.

