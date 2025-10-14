# Replaying, Optimizing and Tuning with Mneme

As described previously, Mneme operates in 3 phases:
1. Phase 1: Build an _recordable_ executable using an LLVM pass, 
2. Phase 2: Recording the trace of the application a user application, and, 
3. Phase 3: Replaying as well as optimizing and tuning the execution of the said application.

Phases 1 and 2 are described in detail on the [Integration with User Codebases](integration.md) page. This page describes Phase 3 in detail. 

## Replaying with Mneme

To replay the execution with Mneme you can use the `replay` tool under `<MNEME-INSTALL-PATH>/bin/replay`. 
The replay tool has the following options that can viewed by passing the `--help` parameter:

```
OVERVIEW: GPU Replay Tool

USAGE: replay [options]

OPTIONS:

Generic Options:

  --help                       - Display available options (--help-hidden for more)
  --help-list                  - Display list of available options (--help-list-hidden for more)
  --version                    - Display the version of this program

Mneme Tool Options:
Mneme CLI options.

  --backend-opt-level=<uint>   - The optimization level to use when optimizing IR
  --middle-opt-level=<char>    - The optimization level to use when optimizing IR
  --mneme-replay-hash=<string> - The Kernel Hash of the Recorded kernels
  --mneme-replay-json=<string> - The json file containing metadata for the recorded kernels
  --repeats=<int>              - The number of repeat executions for every kernel
```

To replay, please provide a path to the `json` file describing a kernel execution as an argument of the `--mneme-replay-json` and specify which runtime instantiation of the kernel you would like to run by specifying the
`mneme-replay-hash` value.

## Optimizing and tuning with Mneme

### Execution example
```shell
mneme execute -db <db-dir> -rid <dynamic-id> \
--db-dir <some-directory> --suffix="some-string" \
--num-trials 1000 -it 2
```

### Tuning example

```shell
# ENABLE JIT:
mneme tune -db <db-dir> -rid <dynamic-id> \
--db-dir <some-directory> --suffix="some-string" \
--num-trials 1000 -it 2 \
--tuner-type optuna \
--search-sampler QMCSampler \
--specialize --prune --internalize

#DISABLE JIT:
mneme tune -db <db-dir> -rid <dynamic-id> \
--db-dir <some-directory> --suffix="some-string" \
--num-trials 1000 -it 2 \
--tuner-type optuna \
--search-sampler QMCSampler \
--prune --internalize
```

### Tuning paramaters
The following three flags allow for tuning and optimization of different application kernels in Mneme. 
1. `--specialize`: This flag helps with enabling and disabling just-in-time compilation (JIT). 
2. `--prune`: Focus on one kernel's tuning by deleting the other kernels from the LLVM Intermediate Representation (IR).  
3. `--internalize`: Make symbols internal. 
