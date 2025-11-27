[![docs (gh-pages)](https://github.com/Olympus-HPC/mneme/actions/workflows/gh-pages-docs.yml/badge.svg)](https://github.com/Olympus-HPC/mneme/actions/workflows/gh-pages-docs.yml)
![License: Apache 2.0 with LLVM exceptions](https://img.shields.io/badge/license-Apache%202.0%20with%20LLVM%20exceptions-blue.svg)

# <img src="docs/assets/MnemeLogoNoText.png" width="128" align="middle" /> Mneme (Μνήμη)

*Named after the Greek goddess of memory, preserves and replays the essence of your application's execution, allowing developers to revisit, analyze, and refine specific moments in code with precision.* 

## Description
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


## Contributions

We welcome all kinds of contributions: new features, bug fixes, documentation edits; it's all great!

To contribute, make a pull request, with `develop` as the destination branch.


# Release

Mneme is released under Apache License (Version 2.0) with LLVM exceptions. For more details, please see the [LICENSE](./LICENSE).

`LLNL-CODE-2000766`

## Citation

If you use this software, please cite it as below:

```bibtex
@inproceedings{parasyris2023scalable,
  title={Scalable Tuning of (OpenMP) GPU Applications via Kernel Record and Replay},
  author={Parasyris, Konstantinos and Georgakoudis, Giorgis and Rangel, Esteban and Laguna, Ignacio and Doerfert, Johannes},
  booktitle={Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
  pages={1--14},
  year={2023}
}
```

