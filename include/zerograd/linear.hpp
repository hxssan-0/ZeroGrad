#pragma once

#include "tensor.hpp"
#include "layer.hpp"
#include <memory>

namespace zerograd
{
    class Linear : public Layer
    {
    private:
        std::shared_ptr<Tensor> weight;
        std::shared_ptr<Tensor> bias;

        std::shared_ptr<Tensor> xavier_uniform_init(
            const std::vector<std::size_t>& shape, 
            std::size_t in_features, 
            std::size_t out_features, 
            bool requires_grad
        );

    public:
        Linear(std::size_t in_features, std::size_t out_features);

        std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;

        std::vector<std::shared_ptr<Tensor>> parameters() override;
    };
}