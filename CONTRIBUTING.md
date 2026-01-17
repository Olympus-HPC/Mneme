# Contributing to Mneme

Thank you for your interest in contributing to Mneme! We welcome contributions of all forms, including bug fixes, new features, documentation improvements, and performance enhancements.

## Development Workflow

1.  **Fork and Clone**: Fork the repository and clone it locally.
2.  **Branch**: Create a feature branch from `develop`.
    ```bash
    git checkout -b feature/my-new-feature develop
    ```
3.  **Implement**: Write your code, adhering to the project's coding style.
4.  **Test**: Ensure your changes pass existing tests and add new tests where appropriate.
    - C++ Tests: Built with CMake (Enable `-DMNEME_ENABLE_TESTS=On`).
    - Python Tests: Run via `pytest`.
5.  **Commit**: Use descriptive commit messages.
6.  **Push and PR**: Push your branch and open a Pull Request against `develop`.

## Coding Style

### C++
We follow the LLVM coding style with slight modifications.
- Use **clang-format** to format your code before submitting. A `.clang-format` file is provided in the root directory.
- Indentation: 2 spaces.
- Headers: Use include guards or `#pragma once`.

### Python
- Use **Black** for formatting.
- Use **Isort** for import sorting.
- Type hints are encouraged for public APIs.
- Configuration is available in `pyproject.toml`.

## Reporting Issues

Please search existing issues before reporting a new one. When opening an issue, provide:
- Mneme version / Commit hash.
- System information (OS, GPU, Driver version).
- Reproduction steps or a minimal example.
