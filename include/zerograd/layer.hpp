#pragma once

#include "tensor.hpp"
#include <memory>
#include <vector>

namespace zerograd
{
    class Layer
    {
    public:
        virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) = 0;

        virtual std::vector<std::shared_ptr<Tensor>> parameters()
        {
            return {}; // no params by default
        }
    
        virtual ~Layer() = default;
    };
}