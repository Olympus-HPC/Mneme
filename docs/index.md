<div style="display: flex; align-items: center; gap: 1.2rem;">

<img src="assets/images/MnemeLogoNotext.svg"
     alt="Mneme logo"
     style="height: 96px;"/>

<div>

*Named after the Greek goddess of memory, Mneme preserves and replays the essence of your application's execution, allowing developers to revisit, analyze, and refine specific moments in code with precision.*

</div>
</div>

## High-Level Overview
[Mneme](https://en.wikipedia.org/wiki/Mneme) is a framework for **recording and replaying GPU kernel executions**
(CUDA / HIP) as **standalone, reproducible executables**.

It enables developers and researchers to:

- capture the full execution context of a GPU kernel,
- replay kernels independently of the original application, and
- experiment with compiler transformations and runtime parameters in isolation.


---

## Execution model

Mneme operates in **three distinct phases**:

### 1. Instrumentation (compile time)

At compile time, the user applies a provided **LLVM instrumentation pass**.
This pass:

- detects GPU global variables and device functions,
- associates them with the corresponding LLVM IR, and
- embeds this information into device memory.

The result is a **recordable executable**.

### 2. Recording (runtime)

The recordable executable is run with representative inputs using the
`mneme record` command.

During recording, Mneme transparently intercepts GPU kernel launches and
captures the required device memory state before execution.
All interception and instrumentation mechanisms are handled internally by Mneme
and do not require changes to the application or runtime environment from the user.

For each recorded kernel, Mneme stores:

- the associated LLVM IR, and
- snapshots of the device memory required to replay the kernel.

At the end of execution, Mneme produces a persistent database of artifacts
(e.g., JSON metadata and memory snapshots) describing each recorded kernel.

---

### 3. Replay and experimentation

In the final phase, kernels can be **replayed as independent executables**.

During replay, users can:

- modify the LLVM IR,
- experiment with compiler transformations,
- autotune kernel launch parameters (e.g., grid and block dimensions), and
- evaluate performance changes in isolation.

---

## Documentation structure

This site contains:

- a **User Guide** covering installation and usage,
- a **Developer Guide** describing internal architecture, and
- API references for both **Python** and **C++** components.

Start with **[Usage → Install](usage/install.md)** to get up and running.

