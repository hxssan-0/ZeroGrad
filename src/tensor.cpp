#include <zerograd/tensor.h>
#include <stdexcept>
#include <algorithm>
#include <iterator>
#include <cmath>

namespace zerograd
{
    Tensor::Tensor(
        std::vector<float> data,
        std::vector<std::size_t> shape,
        bool requires_grad,
        std::vector<std::shared_ptr<Tensor>> _children,
        std::string _op
    ) :
    data(std::move(data)), shape(std::move(shape)), grad(this->data.size(), 0.0f), requires_grad(requires_grad), 
    _children(std::move(_children)), _op(std::move(_op)), strides(this->shape.size(), 0)
    {
        if (!this->shape.empty()) {
            strides.back() = 1;

            for (std::size_t i{this->shape.size() - 1}; i > 0; i--) {
                strides[i - 1] = this->shape[i] * strides[i];
            }
        }
    }

    std::vector<std::size_t> Tensor::compute_broadcast_shape(const std::vector<std::size_t>& shape1, const std::vector<std::size_t>& shape2)
    {
        auto left_shape_it {shape1.rbegin()}, right_shape_it {shape2.rbegin()};
        std::vector<std::size_t> result_shape;

        while (left_shape_it != shape1.rend() || right_shape_it != shape2.rend()) {

            bool left_shape_it_valid = left_shape_it != shape1.rend();
            bool right_shape_it_valid = right_shape_it != shape2.rend();
            
            if (left_shape_it_valid && right_shape_it_valid) {
                std::size_t left_shape = *left_shape_it;
                std::size_t right_shape = *right_shape_it;

                if (left_shape != right_shape && left_shape != 1 && right_shape != 1) {
                    throw std::invalid_argument("ValueError: operands could not be broadcast together");
                }

                result_shape.push_back(std::max(left_shape, right_shape));
            }
            else if (left_shape_it_valid) {
                result_shape.push_back(*left_shape_it);
            }
            else {
                result_shape.push_back(*right_shape_it);
            }

            if (left_shape_it_valid) {
                ++left_shape_it;
            }
            if (right_shape_it_valid) {
                ++right_shape_it;
            }
        }

        std::reverse(result_shape.begin(), result_shape.end());
        return result_shape;
    }

    std::vector<std::size_t> Tensor::pad_shape(const std::vector<std::size_t>& shape, std::size_t result_shape_size)
    {
        std::vector<std::size_t> shape_padded(result_shape_size, 1);

        std::copy(
            shape.begin(), 
            shape.end(), 
            shape_padded.begin() + (result_shape_size - shape.size())
        );

        return shape_padded;
    }

    std::vector<std::size_t> Tensor::compute_padded_strides(const std::vector<std::size_t>& strides, std::size_t result_shape_size, const std::vector<std::size_t>& shape_padded)
    {
        std::vector<std::size_t> strides_padded(result_shape_size, 0);

        std::copy(strides.begin(), strides.end(), strides_padded.begin() + (result_shape_size - strides.size()));

        for (std::size_t i{}; i < strides_padded.size(); ++i) {
            if (shape_padded[i] == 1) {
                strides_padded[i] = 0;
            }
        }

        return strides_padded;
    }

    std::vector<std::size_t> Tensor::convert_flat_to_multi_index(std::size_t flat_idx, const std::vector<std::size_t>& shape)
    {
        std::vector<std::size_t> multi_idx(shape.size());

        for (std::size_t i{shape.size()}; i-- > 0; ) {
            multi_idx[i] = flat_idx % shape[i];
            flat_idx /= shape[i];
        }

        return multi_idx;
    }
    
    std::size_t Tensor::convert_multi_to_flat_index(const std::vector<std::size_t>& multi_idx, const std::vector<std::size_t>& strides)
    {
        std::size_t flat_idx = 0;

        for (std::size_t i{}; i < multi_idx.size(); ++i) {
            flat_idx += multi_idx[i] * strides[i];
        }

        return flat_idx;
    }

    std::size_t Tensor::calculate_total_elements(const std::vector<std::size_t>& shape)
    {
        std::size_t total_elements = 1;

        for (std::size_t dim : shape) {
            total_elements *= dim;
        }

        return total_elements;
    }

    bool Tensor::get_requires_grad() const
    {
        return requires_grad;
    }

    std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& left, const std::shared_ptr<Tensor>& right)
    {
        // first checking if the shapes are compatible and finding out the shape of the result tensor
        std::vector<std::size_t> result_shape = Tensor::compute_broadcast_shape(left->shape, right->shape);

        // calculating total number of elements in the tensor to allocate memory for it
        std::size_t total_elements = Tensor::calculate_total_elements(result_shape);

        std::vector<float> result_data(total_elements, 0.0f);

        // padding the shapes of the tensors
            std::vector<std::size_t> left_shape_padded = 
            (left->shape.size() == result_shape.size()) 
            ? left->shape 
            : Tensor::pad_shape(left->shape, result_shape.size());

            std::vector<std::size_t> right_shape_padded = 
            (right->shape.size() == result_shape.size()) 
            ? right->shape 
            : Tensor::pad_shape(right->shape, result_shape.size());

            // if shapes were padded then we needed padded strides
            std::vector<std::size_t> left_strides_padded = Tensor::compute_padded_strides(
                left->strides, 
                result_shape.size(), 
                left_shape_padded
            );

            std::vector<std::size_t> right_strides_padded = Tensor::compute_padded_strides(
                right->strides, 
                result_shape.size(), 
                right_shape_padded
            );

        // now performing element wise addition
        for (std::size_t i{}; i < total_elements; ++i) {
            std::vector<std::size_t> multi_idx = Tensor::convert_flat_to_multi_index(i, result_shape);
            std::size_t l_idx = Tensor::convert_multi_to_flat_index(multi_idx, left_strides_padded);
            std::size_t r_idx = Tensor::convert_multi_to_flat_index(multi_idx, right_strides_padded);

            result_data[i] = left->data[l_idx] + right->data[r_idx];
        }

        // creating the result Tensor object
        bool requires_grad = left->requires_grad || right->requires_grad;
        std::string op = "+";

        auto result = std::make_shared<Tensor>(
            result_data, 
            result_shape, 
            requires_grad, 
            std::vector<std::shared_ptr<Tensor>>{left, right}, 
            op
        );

        result->_backward = [left, right, out = result.get(), left_strides_padded, right_strides_padded, result_shape = result->shape]() {

            for (std::size_t i{}; i < out->grad.size(); ++i) {

                std::vector<std::size_t> multi_idx = Tensor::convert_flat_to_multi_index(i, result_shape);

                std::size_t l_idx = Tensor::convert_multi_to_flat_index(multi_idx, left_strides_padded);
                std::size_t r_idx = Tensor::convert_multi_to_flat_index(multi_idx, right_strides_padded);

                if (left->requires_grad) {
                    left->grad[l_idx] += out->grad[i];
                }
                if (right->requires_grad) {
                    right->grad[r_idx] += out->grad[i];
                }
            }
        };

        return result;
    }

    std::shared_ptr<Tensor> sub(const std::shared_ptr<Tensor>& left, const std::shared_ptr<Tensor>& right)
    {
        // first checking if the shapes are compatible and finding out the shape of the result tensor
        std::vector<std::size_t> result_shape = Tensor::compute_broadcast_shape(left->shape, right->shape);

        // calculating total number of elements in the tensor to allocate memory for it
        std::size_t total_elements = Tensor::calculate_total_elements(result_shape);

        std::vector<float> result_data(total_elements, 0.0f);

        // padding the shapes of the tensors
            std::vector<std::size_t> left_shape_padded = 
            (left->shape.size() == result_shape.size()) 
            ? left->shape 
            : Tensor::pad_shape(left->shape, result_shape.size());

            std::vector<std::size_t> right_shape_padded = 
            (right->shape.size() == result_shape.size()) 
            ? right->shape 
            : Tensor::pad_shape(right->shape, result_shape.size());

            // if shapes were padded then we needed padded strides
            std::vector<std::size_t> left_strides_padded = Tensor::compute_padded_strides(
                left->strides, 
                result_shape.size(), 
                left_shape_padded
            );

            std::vector<std::size_t> right_strides_padded = Tensor::compute_padded_strides(
                right->strides, 
                result_shape.size(), 
                right_shape_padded
            );

        // now performing element wise subtraction
        for (std::size_t i{}; i < total_elements; ++i) {
            std::vector<std::size_t> multi_idx = Tensor::convert_flat_to_multi_index(i, result_shape);
            std::size_t l_idx = Tensor::convert_multi_to_flat_index(multi_idx, left_strides_padded);
            std::size_t r_idx = Tensor::convert_multi_to_flat_index(multi_idx, right_strides_padded);

            result_data[i] = left->data[l_idx] - right->data[r_idx];
        }

        // creating the result Tensor object
        bool requires_grad = left->requires_grad || right->requires_grad;
        std::string op = "-";

        auto result = std::make_shared<Tensor>(
            result_data, 
            result_shape, 
            requires_grad, 
            std::vector<std::shared_ptr<Tensor>>{left, right}, 
            op
        );

        result->_backward = [left, right, out = result.get(), left_strides_padded, right_strides_padded, result_shape = result->shape]() {

            for (std::size_t i{}; i < out->grad.size(); ++i) {

                std::vector<std::size_t> multi_idx = Tensor::convert_flat_to_multi_index(i, result_shape);

                std::size_t l_idx = Tensor::convert_multi_to_flat_index(multi_idx, left_strides_padded);
                std::size_t r_idx = Tensor::convert_multi_to_flat_index(multi_idx, right_strides_padded);

                if (left->requires_grad) {
                    left->grad[l_idx] += out->grad[i];
                }
                if (right->requires_grad) {
                    right->grad[r_idx] -= out->grad[i];
                }
            }
        };

        return result;
    }

    std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor>& left, const std::shared_ptr<Tensor>& right)
    {
        // first checking if the shapes are compatible and finding out the shape of the result tensor
        std::vector<std::size_t> result_shape = Tensor::compute_broadcast_shape(left->shape, right->shape);

        // calculating total number of elements in the tensor to allocate memory for it
        std::size_t total_elements = Tensor::calculate_total_elements(result_shape);

        std::vector<float> result_data(total_elements, 0.0f);

        // padding the shapes of the tensors
            std::vector<std::size_t> left_shape_padded = 
            (left->shape.size() == result_shape.size()) 
            ? left->shape 
            : Tensor::pad_shape(left->shape, result_shape.size());

            std::vector<std::size_t> right_shape_padded = 
            (right->shape.size() == result_shape.size()) 
            ? right->shape 
            : Tensor::pad_shape(right->shape, result_shape.size());

            // if shapes were padded then we needed padded strides
            std::vector<std::size_t> left_strides_padded = Tensor::compute_padded_strides(
                left->strides, 
                result_shape.size(), 
                left_shape_padded
            );

            std::vector<std::size_t> right_strides_padded = Tensor::compute_padded_strides(
                right->strides, 
                result_shape.size(), 
                right_shape_padded
            );

        // now performing element wise multiplication
        for (std::size_t i{}; i < total_elements; ++i) {
            std::vector<std::size_t> multi_idx = Tensor::convert_flat_to_multi_index(i, result_shape);
            std::size_t l_idx = Tensor::convert_multi_to_flat_index(multi_idx, left_strides_padded);
            std::size_t r_idx = Tensor::convert_multi_to_flat_index(multi_idx, right_strides_padded);

            result_data[i] = left->data[l_idx] * right->data[r_idx];
        }

        // creating the result Tensor object
        bool requires_grad = left->requires_grad || right->requires_grad;
        std::string op = "*";

        auto result = std::make_shared<Tensor>(
            result_data, 
            result_shape, 
            requires_grad, 
            std::vector<std::shared_ptr<Tensor>>{left, right}, 
            op
        );

        result->_backward = [left, right, out = result.get(), left_strides_padded, right_strides_padded, result_shape = result->shape]() {

            for (std::size_t i{}; i < out->grad.size(); ++i) {

                std::vector<std::size_t> multi_idx = Tensor::convert_flat_to_multi_index(i, result_shape);

                std::size_t l_idx = Tensor::convert_multi_to_flat_index(multi_idx, left_strides_padded);
                std::size_t r_idx = Tensor::convert_multi_to_flat_index(multi_idx, right_strides_padded);

                if (left->requires_grad) {
                    left->grad[l_idx] += right->data[i] * out->grad[i];
                }
                if (right->requires_grad) {
                    right->grad[r_idx] += left->data[i] * out->grad[i];
                }
            }
        };

        return result;
    }

    std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& left, const std::shared_ptr<Tensor>& right)
    {
        if (left->shape.size() < 2 || right->shape.size() < 2) {
            throw std::runtime_error("matmul requires tensors to have a minimum rank of 2.");
        }

        // consider left has last 2 dims (M, K) and right has last 2 dims (K, N)
        std::size_t M = left->shape[left->shape.size() - 2];
        std::size_t K_left = left->shape[left->shape.size() - 1];
        std::size_t K_right = right->shape[right->shape.size() - 2];
        std::size_t N = right->shape[right->shape.size() - 1];

        if (K_left != K_right) {
            throw std::runtime_error("dimensions incompatible for matmul.");
        }
        std::size_t K = K_left;

        std::vector<std::size_t> left_batch(left->shape.begin(), left->shape.end() - 2);
        std::vector<std::size_t> right_batch(right->shape.begin(), right->shape.end() - 2);
        std::vector<std::size_t> result_shape = Tensor::compute_broadcast_shape(left_batch, right_batch);
        result_shape.push_back(M); result_shape.push_back(N);

        std::size_t total_elements = Tensor::calculate_total_elements(result_shape);
        std::vector<float> result_data(total_elements, 0.0f);

        std::vector<std::size_t> left_shape_padded = Tensor::pad_shape(left->shape, result_shape.size());
        std::vector<std::size_t> right_shape_padded = Tensor::pad_shape(right->shape, result_shape.size());

        std::vector<std::size_t> left_strides_padded = Tensor::compute_padded_strides(
            left->strides, result_shape.size(), left_shape_padded
        );

        std::vector<std::size_t> right_strides_padded = Tensor::compute_padded_strides(
            right->strides, result_shape.size(), right_shape_padded
        );

        // forward pass
        for (std::size_t i{}; i < total_elements; ++i) {
            std::vector<std::size_t> multi_idx = Tensor::convert_flat_to_multi_index(i, result_shape);
            std::size_t row = multi_idx[result_shape.size() - 2];
            std::size_t col = multi_idx[result_shape.size() - 1];

            float dot_product = 0.0f;
            for (std::size_t k{}; k < K; ++k) {
                auto l_lookup = multi_idx;
                l_lookup[result_shape.size() - 1] = k;

                auto r_lookup = multi_idx;
                r_lookup[result_shape.size() - 2] = k;

                std::size_t l_flat = Tensor::convert_multi_to_flat_index(l_lookup, left_strides_padded);
                std::size_t r_flat = Tensor::convert_multi_to_flat_index(r_lookup, right_strides_padded);

                dot_product += left->data[l_flat] * right->data[r_flat];
            }
            result_data[i] = dot_product;
        }
        
        bool requires_grad = left->requires_grad || right->requires_grad;
        auto result = std::make_shared<Tensor>(
            result_data, 
            result_shape, 
            requires_grad, 
            std::vector<std::shared_ptr<Tensor>>{left, right}, 
            "@"
        );

        result->_backward = [left, right, out = result.get(), left_strides_padded, right_strides_padded, result_shape, K]() {
            for (std::size_t i = 0; i < out->grad.size(); ++i) {
                std::vector<std::size_t> multi_idx = Tensor::convert_flat_to_multi_index(i, result_shape);
                std::size_t row = multi_idx[result_shape.size() - 2];
                std::size_t col = multi_idx[result_shape.size() - 1];

                for (std::size_t k = 0; k < K; ++k) {
                    auto l_lookup = multi_idx;
                    l_lookup[result_shape.size() - 1] = k;

                    auto r_lookup = multi_idx;
                    r_lookup[result_shape.size() - 2] = k;

                    std::size_t l_flat = Tensor::convert_multi_to_flat_index(l_lookup, left_strides_padded);
                    std::size_t r_flat = Tensor::convert_multi_to_flat_index(r_lookup, right_strides_padded);

                    if (left->requires_grad) {
                        left->grad[l_flat] += out->grad[i] * right->data[r_flat];
                    }
                    if (right->requires_grad) {
                        right->grad[r_flat] += left->data[l_flat] * out->grad[i];
                    }
                }
            }
        };

        return result;
    }

    std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor>& tensor)
    {
        std::vector<float> result_data(tensor->data.size());

        for (std::size_t i{}; i < result_data.size(); ++i) {
            result_data[i] = std::max(0.0f, tensor->data[i]);
        }

        auto result = std::make_shared<Tensor>(
            result_data,
            tensor->shape,
            tensor->requires_grad,
            std::vector<std::shared_ptr<Tensor>>{tensor},
            "relu"
        );

        result->_backward = [tensor, out = result.get()]() {
            if (tensor->requires_grad) {
                for (std::size_t i{}; i < out->grad.size(); ++i) {
                    tensor->grad[i] += ((out->data[i] == 0) ? 0 : 1) * out->grad[i];  
                }
            }
        };

        return result;
    }

    float Tensor::compute_sigmoid(float x)
    {
        return 1 / (1 + std::exp(x));
    }

    std::shared_ptr<Tensor> sigmoid(const std::shared_ptr<Tensor>& tensor)
    {
        std::vector<float> result_data(tensor->data.size());

        for (std::size_t i{}; i < result_data.size(); ++i) {
            result_data[i] = Tensor::compute_sigmoid(tensor->data[i]);
        }

        auto result = std::make_shared<Tensor>(
            result_data,
            tensor->shape,
            tensor->requires_grad,
            std::vector<std::shared_ptr<Tensor>>{tensor},
            "sigmoid"
        );

        result->_backward = [tensor, out = result.get()]() {
            if (tensor->requires_grad) {
                for (std::size_t i{}; i < out->grad.size(); ++i) {
                    tensor->grad[i] += (out->data[i] * (1.0f - out->data[i])) * out->grad[i];  
                }
            }
        };

        return result;
    }

    std::shared_ptr<Tensor> tanh(const std::shared_ptr<Tensor>& tensor)
    {
        std::vector<float> result_data(tensor->data.size());

        for (std::size_t i{}; i < result_data.size(); ++i) {
            result_data[i] = std::tanh(tensor->data[i]);
        }

        auto result = std::make_shared<Tensor>(
            result_data,
            tensor->shape,
            tensor->requires_grad,
            std::vector<std::shared_ptr<Tensor>>{tensor},
            "tanh"
        );

        result->_backward = [tensor, out = result.get()]() {
            if (tensor->requires_grad) {
                for (std::size_t i{}; i < out->grad.size(); ++i) {
                    tensor->grad[i] += (1.0f - (out->data[i] * out->data[i])) * out->grad[i];  
                }
            }
        };

        return result;
    }

    std::shared_ptr<Tensor> sum(const std::shared_ptr<Tensor>& tensor)
    {
        float s = 0.0f;

        for (std::size_t i{}; i < tensor->data.size(); ++i) {
            s += tensor->data[i];
        }

        auto result = std::make_shared<Tensor>(
            std::vector<float>{s},
            std::vector<std::size_t>{},
            tensor->requires_grad,
            std::vector<std::shared_ptr<Tensor>>{tensor},
            "sum"
        );

        result->_backward = [tensor, out = result.get()]() {
            if (tensor->requires_grad) {
                for (std::size_t i{}; i < tensor->grad.size(); ++i) {
                    tensor->grad[i] += out->grad[0];  
                }
            }
        };

        return result;
    }

    std::shared_ptr<Tensor> mean(const std::shared_ptr<Tensor>& tensor)
    {
        float m = 0.0f;

        for (std::size_t i{}; i < tensor->data.size(); ++i) {
            m += tensor->data[i];
        }

        m /= static_cast<float>(tensor->data.size());

        auto result = std::make_shared<Tensor>(
            std::vector<float>{m},
            std::vector<std::size_t>{},
            tensor->requires_grad,
            std::vector<std::shared_ptr<Tensor>>{tensor},
            "mean"
        );

        result->_backward = [tensor, out = result.get()]() {
            if (tensor->requires_grad) {
                for (std::size_t i{}; i < tensor->grad.size(); ++i) {
                    tensor->grad[i] += out->grad[0] / static_cast<float>(tensor->grad.size());  
                }
            }
        };

        return result;
    }

    std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& left, const std::shared_ptr<Tensor>& right)
    {
        return add(left, right);
    }

    std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& left, const std::shared_ptr<Tensor>& right)
    {
        return sub(left, right);
    }

    std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& left, const std::shared_ptr<Tensor>& right)
    {
        return mul(left, right);
    }
}