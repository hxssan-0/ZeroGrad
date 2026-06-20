# ZeroGrad 

A custom autograd engine and static memory planner built in C++20 from scratch.

## Project Goals
* Understand the internal mechanics of PyTorch and modern ML frameworks.
* Implement reverse-mode automatic differentiation.
* Build a zero-copy Python bridge using pybind11.
* Optimize heap allocations using a custom memory arena.

## Dev Log
* **[June 2026] Week 0:** Configured WSL 2 environment. Wired up CMake with C++20, Catch2 v3 for testing, and pybind11. Successfully compiled the Linux kernel `perf` tool from source to enable hardware profiling.