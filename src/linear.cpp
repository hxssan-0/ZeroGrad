#include <zerograd/linear.hpp>
#include <cmath>
#include <random>
#include <algorithm>
#include <vector>

namespace zerograd
{
    std::shared_ptr<Tensor> Linear::xavier_uniform_init(
        const std::vector<std::size_t>& shape, 
        std::size_t in_features, 
        std::size_t out_features, 
        bool requires_grad
    )
    {
        float x = std::sqrt(6.0f / static_cast<float>(in_features + out_features));
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

    Linear::Linear(std::size_t in_features, std::size_t out_features)
    {
        // initializing the weight Tensor
        bool weight_requires_grad = true;
        std::vector<std::size_t> weight_shape = {out_features, in_features};

        this->weight = xavier_uniform_init(weight_shape, in_features, out_features, weight_requires_grad);

        // initializing the bias Tensor
        bool bias_requires_grad = true;
        std::vector<std::size_t> bias_shape = {out_features};

        this->bias = xavier_uniform_init(bias_shape, in_features, out_features, bias_requires_grad);

    }

    std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> input)
    {
        return matmul(input, transpose(weight)) + bias;
    }

    std::vector<std::shared_ptr<Tensor>> Linear::parameters()
    {
        return {weight, bias};
    }
}