[![docs (gh-pages)](https://github.com/Olympus-HPC/mneme/actions/workflows/gh-pages-docs.yml/badge.svg)](https://github.com/Olympus-HPC/mneme/actions/workflows/gh-pages-docs.yml)
[![codecov](https://codecov.io/gh/Olympus-HPC/Mneme/graph/badge.svg?token=N8CELEZ277)](https://codecov.io/gh/Olympus-HPC/Mneme)
![License: Apache 2.0 with LLVM exceptions](https://img.shields.io/badge/license-Apache%202.0%20with%20LLVM%20exceptions-blue.svg)

# <img src="docs/assets/images/MnemeLogoNoText.png" width="128" align="middle" /> Mneme (Μνήμη)

*Named after the Greek goddess of memory, preserves and replays the essence of your application's execution, allowing developers to revisit, analyze, and refine specific moments in code with precision.*

[Mneme](https://en.wikipedia.org/wiki/Mneme) is a tool allowing recording the execution of a GPU (CUDA/HIP) kernel and replaying that kernel as an independent executable.

## Documentation

For full usage instructions, tutorials, and API reference, please visit the **[Documentation](https://olympus-hpc.github.io/Mneme/)**.

## Key Features

*   **Record**: Capture GPU kernels from large applications into isolated replayable units.
*   **Replay**: Execute captured kernels independently without the original application context.
*   **Tune**: Optimize kernel parameters (block size, grid size) and compiler passes using Python tools like Optuna.

## Contributions

We welcome all kinds of contributions: new features, bug fixes, documentation edits; it's all great!

To contribute, make a pull request, with `develop` as the destination branch.

## Release

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
