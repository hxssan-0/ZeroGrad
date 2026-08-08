# ZeroGrad

A custom autograd engine and static memory planner built in C++20 from scratch, with Python bindings.

## Goals

1. Core autograd engine (`Scalar`/`Tensor`, reverse-mode autodiff) **COMPLETE**
2. CNN support (Conv2d, BatchNorm, pooling) **COMPLETE**
3. Arena allocator + static memory planner
4. Benchmarks vs PyTorch + report

## Status

Phase 1 complete : MNIST trained in both C++ ad Python with a 97%+ accuracy.

## Build

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

## Test

```bash
./zerograd_tests
```

## Train (MLP)

```bash
./train_mnist
PYTHONPATH=build/python python3 python/zerograd/train_mnist.py
```

## Train (CNN)

```bash
./train_mnist_cnn
```