#include <zerograd/tensor.h>
#include <stdexcept>
#include <algorithm>
#include <iterator>

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