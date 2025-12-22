<div style="display: flex; align-items: center; gap: 1.2rem;">

<img src="assets/images/MnemeLogoNotext.svg"
     alt="Mneme logo"
     style="height: 96px;"/>

<div>

*Named after the Greek goddess of memory, Mneme preserves and replays the essence of your application's execution, allowing developers to revisit, analyze, and refine specific moments in code with precision.*

</div>
</div>

## High-Level Overview
[Mneme](https://en.wikipedia.org/wiki/Mneme) is a tool allowing recording the execution of a GPU (CUDA/HIP) kernel and replaying that kernel as an independent executable.

Mneme operates in 3 phases. First, during compile time, the user needs to apply a provided LLVM pass to instrument the code. This pass detects the global variables
and functions on the GPU device, and stores this information with the respective LLVM-IR in the global device memory. The compilation generates a _recordable_ executable. 

The second phase involves running the _recordable_ executable with a desired input and using `LD_PRELOAD` to enable recording. When recording, before invoking a device kernel, 
the pre-loaded library stores device memory in persistent storage and associates the memory with the device kernel and an LLVM IR file. At the end of the recorded execution,
the pre-loaded library generates a database in the form of a collection of `json` files, each containing information regarding the LLVM-IR files and the snapshots of device memory for a single GPU kernel.

During the third and last phase, the user can replay the execution of a kernel as a separate independent executable. In addition to executing the kernel, the user can also modify the LLVM IR file and
auto-tune parameters such as kernel launch-bounds or kernel runtime execution parameters (e.g. Kernel Block and Grid Dimensions).

This documentation contains the user guide and  developers' manual for
[Mneme](https://github.com/Olympus-HPC/Mneme).

