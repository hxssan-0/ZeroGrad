#pragma once

#include "layer.h"
#include "tensor.h"

namespace zerograd
{
    class ReLU : public Layer
    {
    public:
        std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override
        {
            return relu(input);
        }
    };

    class Sigmoid : public Layer
    {
    public:
        std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override
        {
            return sigmoid(input);
        }
    };

    class Tanh : public Layer
    {
    public:
        std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override
        {
            return tanh(input);
        }
    };
}
