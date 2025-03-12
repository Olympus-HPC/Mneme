# The Mneme documentation

# Mneme (Μνήμη)  
*Named after the Greek goddess of memory, preserves and replays the essence of your application's execution, allowing developers to revisit, analyze, and refine specific moments in code with precision.* 

## Descritpion
"[Mneme](https://en.wikipedia.org/wiki/Mneme)" is a tool allowing recording the execution of a GPU (CUDA/HIP) kernel and replaying that kernel as an independent executable.

The tool operates in 3 phases. During compile time the user needs to apply a provided LLVM pass to instrument the code. The pass detects all device global variables
and device functions and stores this information with the respective LLVM-IR in the global device memory. The compilation generates a `record-able` executable. 

The second phase involves running the application executable with a desired input and using `LD_PRELOAD` to enable recording. When recording before invoking a device kernel 
the pre-loaded library stores device memory in persistent storage and associates the memory with the device kernel and an LLVM IR file. At the end of the recorded execution
the pre-load library generates a database in the form of a `JSON` file containing information regarding the LLVM-IR files and the snapshots of device memory. 

During the third and last phase the user can replay the execution of an kernel as a separate independent executable. Besides executing it the user can modify the LLVM IR file and
auto-tune parameters such as kernel launch-bounds or kernel runtime execution parameters (e.g. Kernel Block and Grid Dimensions).

This documentation contains the user guide and  developers' manual for
[Mneme](https://github.com/Olympus-HPC/Mneme).

