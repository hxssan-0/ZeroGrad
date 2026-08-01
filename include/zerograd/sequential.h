#pragma once

#include "layer.h"
#include "tensor.h"
#include <memory>
#include <vector>

namespace zerograd
{
    class Sequential
    {
    private:
        std::vector<std::shared_ptr<Layer>> layers;

    public:
        std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input);

        std::vector<std::shared_ptr<Tensor>> parameters();

        void add(std::shared_ptr<Layer> layer);
    };
}