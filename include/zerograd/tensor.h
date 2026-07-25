#pragma once

#include <memory>
#include <vector>
#include <cstddef>
#include <functional>
#include <string>
#include <unordered_set>

namespace zerograd
{
    class Tensor : public std::enable_shared_from_this<Tensor>
    {
    private:
        bool requires_grad{}; // false by default to mimic PyTorch
        std::function<void()> _backward = [](){};
        std::vector<std::shared_ptr<Tensor>> _children;
        std::string _op;

        static std::vector<std::size_t> compute_broadcast_shape(const std::vector<std::size_t>& shape1, const std::vector<std::size_t>& shape2);
        static std::vector<std::size_t> pad_shape(const std::vector<std::size_t>& shape, std::size_t result_shape_size);
        static std::vector<std::size_t> compute_padded_strides(const std::vector<std::size_t>& strides, std::size_t result_shape_size, const std::vector<std::size_t>& shape_padded);
        static std::vector<std::size_t> convert_flat_to_multi_index(std::size_t flat_idx, const std::vector<std::size_t>& shape);
        static std::size_t convert_multi_to_flat_index(const std::vector<std::size_t>& multi_idx, const std::vector<std::size_t>& strides);

        static float compute_sigmoid(float x);

        static void build_topo(
            const std::shared_ptr<Tensor>& node,
            std::vector<std::shared_ptr<Tensor>>& topo,
            std::unordered_set<std::shared_ptr<Tensor>>& visited
        );

    public:
        std::vector<float> data;
        std::vector<std::size_t> shape;
        std::vector<std::size_t> strides;
        std::vector<float> grad;

        static std::size_t calculate_total_elements(const std::vector<std::size_t>& shape);

        explicit Tensor(
            std::vector<float> data,
            std::vector<std::size_t> shape,
            bool requires_grad = false,
            std::vector<std::shared_ptr<Tensor>> _children = {},
            std::string _op = ""
        );

        bool get_requires_grad() const;

        friend std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& left, const std::shared_ptr<Tensor>& right);
        friend std::shared_ptr<Tensor> sub(const std::shared_ptr<Tensor>& left, const std::shared_ptr<Tensor>& right);
        friend std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor>& left, const std::shared_ptr<Tensor>& right);
        friend std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& left, const std::shared_ptr<Tensor>& right);
        friend std::shared_ptr<Tensor> log(const std::shared_ptr<Tensor>& tensor);
        friend std::shared_ptr<Tensor> exp(const std::shared_ptr<Tensor>& tensor);

        friend std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor>& tensor);
        friend std::shared_ptr<Tensor> sigmoid(const std::shared_ptr<Tensor>& tensor);
        friend std::shared_ptr<Tensor> tanh(const std::shared_ptr<Tensor>& tensor);

        friend std::shared_ptr<Tensor> sum(const std::shared_ptr<Tensor>& tensor);
        friend std::shared_ptr<Tensor> mean(const std::shared_ptr<Tensor>& tensor);

        friend std::shared_ptr<Tensor> transpose(const std::shared_ptr<Tensor>& tensor);

        friend std::shared_ptr<Tensor> max(const std::shared_ptr<Tensor>& tensor);

        void backward();

        friend std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& left, const std::shared_ptr<Tensor>& right);
        friend std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& left, const std::shared_ptr<Tensor>& right);
        friend std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& left, const std::shared_ptr<Tensor>& right);

        friend std::shared_ptr<Tensor> mse_loss(const std::shared_ptr<Tensor>& prediction, const std::shared_ptr<Tensor>& target);
        friend std::shared_ptr<Tensor> ce_loss(const std::shared_ptr<Tensor>& logits, const std::vector<std::size_t>& target_classes);
    };
}

