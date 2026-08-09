#include <zerograd/hidden.hpp>
#include <cmath>
#include <random>
#include <algorithm>
#include <vector>

namespace zerograd
{  
    // Conv2d layer
    std::shared_ptr<Tensor> Conv2d::xavier_uniform_init(
        const std::vector<std::size_t>& shape,
        std::size_t fan_in,
        std::size_t fan_out,
        bool requires_grad
    )
    {
        float x = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-x, x);

        std::size_t total_elements = Tensor::calculate_total_elements(shape);
        std::vector<float> data(total_elements);
        std::generate(data.begin(), data.end(), [&]() {
            return dist(gen);
        });

        return std::make_shared<Tensor>(data, shape, requires_grad);
    }

    Conv2d::Conv2d(std::size_t in_channels, std::size_t out_channels, 
        std::size_t kernel_h, std::size_t kernel_w,
        std::size_t stride_, std::size_t padding_) : stride(stride_), padding(padding_)
    {
        std::size_t fan_in = in_channels * kernel_h * kernel_w;
        std::size_t fan_out = out_channels * kernel_h * kernel_w;

        bool weight_requires_grad = true;
        std::vector<std::size_t> weight_shape = {out_channels, in_channels, kernel_h, kernel_w};
        weight = xavier_uniform_init(weight_shape, fan_in, fan_out, weight_requires_grad); 

        bool bias_requires_grad = true;
        std::vector<std::size_t> bias_shape = {out_channels};
        std::vector<float> bias_data(out_channels, 0.0f);
        bias = std::make_shared<Tensor>(bias_data, bias_shape, bias_requires_grad);
    }

    std::shared_ptr<Tensor> Conv2d::forward(std::shared_ptr<Tensor> input)
    {
        return zerograd::conv2d(input, weight, bias, stride, padding);
    }

    std::vector<std::shared_ptr<Tensor>> Conv2d::parameters()
    {
        return {weight, bias};
    }

    // MaxPool2d layer
    MaxPool2d::MaxPool2d(std::size_t kernel_h_, std::size_t kernel_w_,
           std::size_t stride_, std::size_t padding_) : kernel_h(kernel_h_), kernel_w(kernel_w_), stride(stride_), padding(padding_)
    {
    }

    std::shared_ptr<Tensor> MaxPool2d::forward(std::shared_ptr<Tensor> input)
    {
        return zerograd::maxPool2d(input, kernel_h, kernel_w, stride, padding);
    }

    // Flatten layer
    std::shared_ptr<Tensor> Flatten::forward(std::shared_ptr<Tensor> input)
    {
        return zerograd::flatten(input);
    }
}