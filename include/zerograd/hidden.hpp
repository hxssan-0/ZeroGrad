#pragma once

#include "layer.hpp"
#include <memory>

namespace zerograd
{
    class Conv2d : public Layer
    {
    private:
        std::shared_ptr<Tensor> weight;
        std::shared_ptr<Tensor> bias;
        std::size_t stride, padding;

        std::shared_ptr<Tensor> xavier_uniform_init(
            const std::vector<std::size_t>& shape,
            std::size_t fan_in,
            std::size_t fan_out,
            bool requires_grad
        );

    public:
        Conv2d(std::size_t in_channels, std::size_t out_channels, 
               std::size_t kernel_h, std::size_t kernel_w,
               std::size_t stride_ = 1, std::size_t padding_ = 0);

        std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;

        std::vector<std::shared_ptr<Tensor>> parameters() override;
    };

    class MaxPool2d : public Layer
    {
    private:
        std::size_t stride, padding, kernel_h, kernel_w;

    public:
        MaxPool2d(std::size_t kernel_h_, std::size_t kernel_w_,
               std::size_t stride_ = 1, std::size_t padding_ = 0);

        std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    };

    class Flatten : public Layer
    {
    public:
        Flatten() = default;
        
        std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    };
}