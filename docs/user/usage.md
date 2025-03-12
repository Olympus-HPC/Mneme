# User Interface

Mneme operates in 3 phases:
1. Build an executable using the mneme pass as discussed in [integration](./integration.md), 
2. Record the execution of some executable.
3. Replay the execution of the said executable.

## Record an execution with Mneme

To record a binary execution (e.g. `hpc-application`) instrumented with the mneme pass you can execute:

```
LD_PRELOAD=<MNEME-INSTALL-PATH>/lib64/librecord.so hpc-application <args>
```

When running this command Mneme will record all kernel invocations of the applicatios and will store their memory and application code
under the current directory. 

To modify the storage directory check the [environment variables](config.md).


## Replay an execution with Mneme

To replay the execution with Mneme you can use the replay tool under `<MNEME-INSTALL-PATH>/bin/replay`. 
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

To replay please provide a path to the json file describing a kernel execution as an argument of the `--mneme-replay-json` and specify which runtime instantiation of the kernel you would like to run by specifying the
`mneme-replay-hash` value.
