# ZeroGrad 

A custom autograd engine and static memory planner built in C++20 from scratch.

## Project Goals
For now, following are the major goals of the project:

1. To implement Scalar and Tensor classes and be able to run neural network training loops using
   gradient descent.
2. To make the engine richer with the ability to work on convolutional neural networks.
3. To implement an arena allocator and experiment with different static memory planning algorithms.