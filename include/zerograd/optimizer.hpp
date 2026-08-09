#pragma once

#include "tensor.hpp"
#include <vector>
#include <memory>

namespace zerograd
{
    class Optimizer
    {
    private:
        std::vector<std::shared_ptr<Tensor>> parameters;
        float lr; // learning rate

    public:
        Optimizer(std::vector<std::shared_ptr<Tensor>> parameters, float lr);

        void zero_grad();

        void step();
    };
}