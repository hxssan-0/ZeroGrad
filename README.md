# ZeroGrad

<p align="center">
  <img src="https://skillicons.dev/icons?i=cpp,python,cmake" />
  <img src="https://img.shields.io/badge/pybind11-Bindings-FFB13B?style=for-the-badge">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/Valgrind-000000?style=for-the-badge">
  <img src="https://img.shields.io/badge/Catch2-Testing-green?style=for-the-badge">
</p>

<p align="center">
A static memory planner and autograd engine written from scratch in <b>C++20</b> with Python bindings via <b>pybind11</b>.
</p>

---

## Overview

ZeroGrad is a deep learning framework that eliminates runtime memory allocation overhead during neural network training.

Instead of relying on dynamic allocation or runtime caching allocators, ZeroGrad performs **offline computational graph analysis** to determine tensor lifetimes, allowing physical memory to be reused inside a single preallocated arena.

This enables deterministic memory layouts with **zero runtime heap allocations** during training.

---

# Features

## Autograd Engine

- Reverse-mode automatic differentiation
- Computational DAG construction
- Automatic topological sorting
- Gradient propagation
- Scalar & tensor operations
    - Addition
    - Subtraction
    - Multiplication
    - Matrix multiplication
    - Log
    - Exp
    - Sum
    - Mean
    - Max
    - Transpose
    - Softmax

---

## Loss Functions

- Mean Squared Error Loss
- Cross Entropy Loss

---

## Neural Network Layers

### Core Layers

- Linear
- Conv2d
- MaxPool2d
- Flatten

### Activation Functions

- ReLU
- Sigmoid
- Tanh

### Containers

- Sequential model builder
- Automatic parameter aggregation

---

## Training Pipeline

- Native MNIST IDX parser
- Mini-batch DataLoader
- Dataset shuffling
- Optimizer
    - SGD
    - zero_grad()
    - step()
- Loss Functions
    - Cross Entropy
    - Mean Squared Error
- Accuracy Metrics

---

## Static Graph Analysis

### GraphValidator

- 3-state DFS cycle detection
- Ensures computation graphs remain valid DAGs

### GraphAnalyzer

Offline `dry_forward()` execution that records

- tensor birth step
- tensor death step
- tensor size
- lifetime intervals

### GraphSerializer

Exports

- computational graph
- tensor metadata
- edges
- shapes
- lifetime intervals

to JSON for visualization and profiling.

---

## Python Bindings

Entire C++ backend exposed through **pybind11**, providing feature parity between

- native C++ training
- Python training

via `_zerograd_backend`.

---

# Roadmap

| Phase | Description | Status |
|------|-------------|--------|
| Phase 1 | Core Autograd Engine | Complete |
| Phase 2 | Extended CNN Support and Computation Graph Analysis | Complete |
| Phase 3 | Static Memory Planning | WIP |
| Phase 4 | Final Benchmarks and Report  | TODO |

---

# Build

## Requirements

### Core Dependencies

- **C++20** compatible compiler
- **Python 3.8+**
- **CMake 3.18+**
- **pybind11**

### Development & Benchmarking

- **Catch2**: C++ unit testing framework
- **PyTorch**: baseline implementation for performance comparison
- **Valgrind Memcheck**: heap allocation profiling

---

## Clone

```bash
git clone https://github.com/hxssan-0/ZeroGrad.git

cd zerograd
```

## Build

```bash
mkdir build

cd build

cmake -DCMAKE_BUILD_TYPE=Release ..

cmake --build . -j$(nproc)
```

---

# Tests

```bash
cd build
./zerograd_tests
```

---

# Training

## MLP

### C++

```bash
cd build
./train_mnist
```

### Python

```bash
cd python/zerograd
PYTHONPATH=../../build/python python3 python/zerograd/train_mnist.py
```

---

## CNN

### C++

```bash
cd build
./train_mnist_cnn
```

### Python

```bash
cd python/zerograd
PYTHONPATH=../../build/python python3 python/zerograd/train_mnist_cnn.py
```

---

# PyTorch Baseline

To compare against PyTorch's runtime allocator, ZeroGrad measures heap allocations using **Valgrind Memcheck**.

Python startup introduces many transient allocations, so we compute a delta between warmup and training iterations.

## Baseline Profiling

```bash
# Initialization + 3 warmup iterations

valgrind ./venv/bin/python python/pytorch/baseline_0.py 2>&1 | grep "allocs"

# Initialization + 3 warmup + 10 training iterations

valgrind ./venv/bin/python python/pytorch/baseline_10.py 2>&1 | grep "allocs"
```

---

## Allocation Formula

Heap Allocations / Step

```
(Allocations10 − Allocations0) / 10
```

Bytes / Step

```
(Bytes10 − Bytes0) / 10
```

---