#include <zerograd/optimizer.h>

#include <algorithm>
#include <cstddef>

namespace zerograd
{
    Optimizer::Optimizer(std::vector<std::shared_ptr<Tensor>> parameters, float lr)
    : parameters(std::move(parameters)), lr(lr)
    {
    }

    void Optimizer::zero_grad()
    {
        for (auto& parameter : parameters) {
            std::ranges::fill(parameter->grad, 0.0f);
        }
    }

    void Optimizer::step()
    {
        for (auto& parameter : parameters) {
            for (std::size_t i{}; i < parameter->data.size(); ++i) {
                parameter->data[i] -= lr * parameter->grad[i];
            }
        }
    }
}