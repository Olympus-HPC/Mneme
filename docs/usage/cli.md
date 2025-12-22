# Mneme Command line Interface

## mneme record

```bash
mneme record [options] -- <command> [args]
```
### Description
The mneme record command executes an application while recording GPU
kernel executions and the associated device memory state.

Recording is performed transparently by Mneme and does not require
modifying the application source code or runtime environment.
Each recorded kernel execution is stored as an independent replayable
artifact.

### Arguments

| Argument | Description                                                   |
| -------- | ------------------------------------------------------------- |
| `cmd`    | The executable to run, followed by its command-line arguments |


| Option                                  | Description                                                                                                               |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------            |
| `-rdb`, `--record-db-dir`               | Path to a **existing** directory where recorded artifacts (metadata, LLVM IR, and device memory snapshots) will be stored |
| `-vass`, `--virtual-address-space-size` | Size (in **GB**) of the virtual address space allocated by Mneme for recording                                            |
| `-mr`, `--per-kernel-max-recordings`    | Maximum number of times the same GPU kernel may be recorded with different dynamic hashes                                 |
| `-h`, `--help`                          | Show help message and exit                                                                                                |

### Recording database

Each invocation of mneme record produces a recording database for every kernel executed by the `command`. The kernel databases are stored
under the directory specified by `--record-db-dir`.

The database contains:
- a JSON metadata file describing the recorded kernel(s),
- the associated LLVM IR used for replay, and
- device memory snapshots captured before and after kernel execution.

See Usage → Artifacts for a detailed description of the database layout.

### Examples

1. Record a GPU application execution:
```bash
mneme record -rdb record-dir -- ./vecAdd 1024
```

2. Limit the number of recordings per kernel:
```bash
mneme record -rdb record-dir -mr 1 -- ./vecAdd 1024
```

3. Increase the virtual address space used during recording:

```bash
mneme record -rdb record-dir -vass 64 -- ./vecAdd 1024
```

### Notes

!!! note
    Mneme may record multiple instances of the same kernel if it is invoked with different dynamic execution contexts. The `--per-kernel-max-recordings` option can be used to limit this behavior.

!!! note
    The virtual address space size should be chosen large enough to accommodate all device allocations performed by the application during kernel execution

### Exit status

0 on successful recording non-zero if execution or recording fails

## mneme replay

```bash
mneme replay [options] <passes>
```

### Description

The `mneme replay` command replays a previously recorded GPU kernel
execution as an independent executable.

During replay, users may modify kernel launch parameters, enable or
disable specialization, adjust compiler optimization levels, and
experiment with alternative code-generation strategies.
Replay executes the kernel using the recorded device memory state and
verifies correctness against the original execution.

### Arguments

| Argument | Description                                                                       |
| -------- | --------------------------------------------------------------------------------- |
| `passes` | Compilation pipeline used to compile and execute the kernel (e.g., `default<O3>`) |


### Required Options

| Option                      | Description                                          |
| --------------------------- | ---------------------------------------------------- |
| `-rdb`, `--record-database` | Path to the Mneme JSON recording database file       |
| `-record-id`, `-rid`        | Identifier of the recorded kernel instance to replay |

### Kernel launch configuration
When omitted, all values default to those recorded during execution.

| Option                   | Description                 |
| ------------------------ | --------------------------- |
| `--grid-dim-x`, `-gidx`  | Override `GridDim.x`        |
| `--grid-dim-y`, `-gidy`  | Override `GridDim.y`        |
| `--grid-dim-z`, `-gidz`  | Override `GridDim.z`        |
| `--block-dim-x`, `-bidx` | Override `BlockDim.x`       |
| `--block-dim-y`, `-bidy` | Override `BlockDim.y`       |
| `--block-dim-z`, `-bidz` | Override `BlockDim.z`       |
| `--shared-mem`, `-shem`  | Override shared memory size |

### Specialize Options

| Option                                                  | Description                                                           |
| ------------------------------------------------------- | --------------------------------------------------------------------- |
| `--specialize`, `--no-specialize`                       | Enable or disable argument specialization (default: disabled)         |
| `--specialize-dims`, `--no-specialize-dims`, `-sdims`   | Specialize `ThreadID.*`, `BlockDim.*`, and `GridDim.*` with constants |
| `--set-launch-bounds`, `--no-set-launch-bounds`, `-slb` | Enable or disable kernel launch bounds                                |
| `--max-threads`                                         | Set launch-bound `max_threads` to the provided value                  |
| `--min-threads-per-block`                               | Set launch-bound `min_blocks_per_sm`                                  |

### Code generation Options

| Option                    | Description                                                    |
| ------------------------- | -------------------------------------------------------------- |
| `--codegen-opt`, `-co`    | Backend optimization level used during machine code generation |
| `--codegen-method`, `-cm` | Code-generation backend to use (e.g., `serial`)                |
| `--output-ll`, `-ol`      | Store the generated LLVM IR to the specified file              |

### Execution Options

| Option                | Description                                                           |
| --------------------- | --------------------------------------------------------------------- |
| `--iterations`, `-it` | Number of iterations to execute the kernel for statistical evaluation |


### Examples

1. Replay a recorded kernel with default settings

```bash
mneme replay \
  -rdb record-dir/15941914485064662553.json \
  -rid 16313427880266313990 \
  "default<O3>"
```

2. Replay while overriding launch configuration

```bash
mneme replay \
  -rdb record-dir/15941914485064662553.json \
  -rid 16313427880266313990 \
  --block-dim-x 256 \
  --grid-dim-x 40000 \
  "default<O3>"
```

3. Replay with specialization and multiple iterations: 

```bash
mneme replay \
  -rdb record-dir/15941914485064662553.json \
  -rid 16313427880266313990 \
  --specialize \
  --iterations 10 \
  "default<O3>"
```

### Notes
!!! note
    When launch parameters are not explicitly provided, Mneme reuses the
    values recorded during the original execution.

!!! note
    Enabling specialization or modifying launch bounds may change kernel
    performance characteristics and should be validated carefully.

### Exit status

0 on successful replay non-zero if replay or compilation fails

## mneme config

```bash
mneme config <key>
```

### Description

The mneme config command queries build and toolchain configuration
information required to compile and link applications against Mneme.

This command is primarily intended to be used from build systems
(e.g., CMake) and scripts to ensure consistent compiler and library
configuration.

### Arguments

| Key          | Description                                                   |
| ------------ | ------------------------------------------------------------- |
| `cc`         | C compiler to be used when building Mneme-instrumented code   |
| `cxx`        | C++ compiler to be used when building Mneme-instrumented code |
| `prefix`     | Installation prefix of the Mneme package                      |
| `libdir`     | Directory containing Mneme libraries                          |
| `includedir` | Directory containing Mneme headers                            |
| `cmakedir`   | Directory containing Mneme CMake configuration files          |
| `cflags`     | Compiler flags required to build against Mneme                |
| `ldflags`    | Linker flags required to link against Mneme                   |

| Option         | Description                |
| -------------- | -------------------------- |
| `-h`, `--help` | Show help message and exit |

### Examples

1. Query the C compiler used by Mneme:
```bash
mneme config cc
```

2. Query the C++ compiler:

```
mneme config cxx
```

3. Obtain the CMake configuration directory for use in a build system:
```bash
mneme config cmakedir
```


4. Use Mneme-provided compilers in a CMake invocation:

```bash
cmake -DCMAKE_C_COMPILER=$(mneme config cc) \
      -DCMAKE_CXX_COMPILER=$(mneme config cxx) \
      -DCMAKE_PREFIX_PATH=$(mneme config cmakedir) \
      ..
```

### Notes

!!! note
    The values returned by mneme config are guaranteed to be consistent
    with the Mneme installation and should be preferred over manually
    specifying compiler paths or flags.

!!! note
    This command performs no side effects and may be safely invoked
    multiple times from scripts or build systems.

### Exit status

0 on successful replay non-zero if replay or compilation fails


## mneme clean

```bash
mneme clean <record_database> [record_database ...]
```

### Description

The mneme clean command removes recorded artifacts associated with one
or more Mneme recording databases.

This command is intended to help users reclaim disk space by deleting
device memory snapshots and auxiliary files that are no longer needed
after experimentation or tuning.

### Arguments
| Argument          | Description                                                    |
| ----------------- | -------------------------------------------------------------- |
| `record_database` | One or more Mneme recording database files (`*.json`) to clean |

### Options

| Option         | Description                |
| -------------- | -------------------------- |
| `-h`, `--help` | Show help message and exit |

### Behavior

For each provided recording database, `mneme clean`:
1. identifies associated recorded artifacts (e.g., memory snapshots and IR files), and
2. removes those artifacts from disk.

The recording database file itself will be removed.

### Examples

1. Clean a single recording database:
```bash
mneme clean record-example-dir/15941914485064662553.json
```

2. Clean multiple recording databases at once:
```bash
mneme clean record-example-dir/*.json
```

### Notes

!!! note
    This operation is destructive. Removed artifacts cannot be recovered
    once deleted.

!!! note
    The mneme clean command does not require access to the original
    executable or compiler toolchain.

### Exit status
0 on successful cleanup non-zero if one or more databases cannot be processed


## mneme move

```bash
mneme move <record_database> [record_database ...]
```

### Description

The `mneme move` command relocates one or more Mneme recording databases
and their associated artifacts to a new destination.

This command is useful for reorganizing recordings, migrating data to
different storage locations, or consolidating multiple recordings into
a common directory.

### Arguments

| Argument          | Description                                                                               |
| ----------------- | ----------------------------------------------------------------------------------------- |
| `record_database` | Source paths of Mneme recording database files (`*.json`), followed by a destination path |

The final argument specifies the destination directory to which the
recording databases and all associated artifacts will be moved.

### Options

| Option         | Description                |
| -------------- | -------------------------- |
| `-h`, `--help` | Show help message and exit |


### Behavior

For each provided recording database, mneme move:

relocates the JSON recording database file, and

moves all associated artifacts (e.g., memory snapshots and LLVM IR files)
to the specified destination.

Internal references within the recording database are updated to reflect
the new location.

### Examples

Move a single recording database to a new directory:

```bash
mneme move record-example-dir/15941914485064662553.json archive/
```

Move multiple recording databases to a common destination:

```bash
mneme move record-example-dir/*.json archive/
```

### Notes

!!! note
    The destination path must exist and be writable.

!!! note
    The move operation preserves the internal consistency of the
    recording database and does not require re-recording.

### Exit status

0 on successful move non-zero if one or more databases cannot be moved


## mneme copy

```bash
mneme copy <record_database> [record_database ...]
```

### Description

The mneme copy command creates copies of one or more Mneme recording
databases and their associated artifacts at a specified destination.

This command is useful for archiving recordings, creating backups, or
duplicating datasets for experimentation without modifying the original
recordings.

### Arguments

| Argument          | Description                                                                               |
| ----------------- | ----------------------------------------------------------------------------------------- |
| `record_database` | Source paths of Mneme recording database files (`*.json`), followed by a destination path |

The final argument specifies the destination directory to which the
recording databases and all associated artifacts will be copied.

### Options

| Option         | Description                |
| -------------- | -------------------------- |
| `-h`, `--help` | Show help message and exit |

### Behavior

For each provided recording database, mneme copy:
1. copies the JSON recording database file, and
2. duplicates all associated artifacts (e.g., device memory snapshots and LLVM IR files)
into the destination directory.

The original recording databases and artifacts remain unchanged.


### Examples

1. Copy a single recording database to a new directory:

```bash
mneme copy record-example-dir/15941914485064662553.json backup/
```

2. Copy multiple recording databases at once:

```bash
mneme copy record-example-dir/*.json backup/
```

### Notes

!!! note
    The destination path must exist and be writable.

!!! note
    Unlike mneme move, the mneme copy command leaves the original
    recording databases and artifacts intact.
