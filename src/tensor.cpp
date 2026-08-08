#include <zerograd/tensor.h>
#include <stdexcept>
#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <limits>
#include <functional>
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
            throw std::runtime_error("Matmul requires tensors of rank >= 2.");
        }

        std::size_t left_rank = left->shape.size();
        std::size_t right_rank = right->shape.size();

        std::size_t M = left->shape[left_rank - 2];
        std::size_t K = left->shape[left_rank - 1];
        std::size_t N = right->shape[right_rank - 1];

        if (right->shape[right_rank - 2] != K) {
            throw std::runtime_error("Incompatible matrix dimensions for matmul.");
        }

        // 1. Separate batch dimensions from 2D matrix dimensions
        std::vector<std::size_t> left_batch(left->shape.begin(), left->shape.end() - 2);
        std::vector<std::size_t> right_batch(right->shape.begin(), right->shape.end() - 2);
        std::vector<std::size_t> res_batch = Tensor::compute_broadcast_shape(left_batch, right_batch);

        std::vector<std::size_t> result_shape = res_batch;
        result_shape.push_back(M);
        result_shape.push_back(N);

        std::size_t num_batches = Tensor::calculate_total_elements(res_batch);
        if (res_batch.empty()) num_batches = 1;

        std::size_t total_elements = num_batches * M * N;
        std::vector<float> result_data(total_elements, 0.0f);

        // 2. Compute strides for batch and 2D components
        std::vector<std::size_t> left_padded_shape = Tensor::pad_shape(left->shape, result_shape.size());
        std::vector<std::size_t> right_padded_shape = Tensor::pad_shape(right->shape, result_shape.size());

        std::vector<std::size_t> left_padded_strides = Tensor::compute_padded_strides(
            left->strides, result_shape.size(), left_padded_shape
        );
        std::vector<std::size_t> right_padded_strides = Tensor::compute_padded_strides(
            right->strides, result_shape.size(), right_padded_shape
        );

        std::size_t l_stride_M = left_padded_strides[result_shape.size() - 2];
        std::size_t l_stride_K = left_padded_strides[result_shape.size() - 1];
        std::size_t r_stride_K = right_padded_strides[result_shape.size() - 2];
        std::size_t r_stride_N = right_padded_strides[result_shape.size() - 1];

        for (std::size_t b = 0; b < num_batches; ++b) {
            std::size_t l_batch_offset = 0;
            std::size_t r_batch_offset = 0;

            if (!res_batch.empty()) {
                std::vector<std::size_t> batch_multi_idx = Tensor::convert_flat_to_multi_index(b, res_batch);
                for (std::size_t dim = 0; dim < res_batch.size(); ++dim) {
                    l_batch_offset += batch_multi_idx[dim] * left_padded_strides[dim];
                    r_batch_offset += batch_multi_idx[dim] * right_padded_strides[dim];
                }
            }

            std::size_t out_batch_offset = b * M * N;

            for (std::size_t i = 0; i < M; ++i) {
                std::size_t l_row_offset = l_batch_offset + i * l_stride_M;
                std::size_t out_row_offset = out_batch_offset + i * N;

                for (std::size_t j = 0; j < N; ++j) {
                    std::size_t r_col_offset = r_batch_offset + j * r_stride_N;
                    float dot_product = 0.0f;

                    for (std::size_t k = 0; k < K; ++k) {
                        dot_product += left->data[l_row_offset + k * l_stride_K] * 
                                    right->data[r_col_offset + k * r_stride_K];
                    }
                    result_data[out_row_offset + j] = dot_product;
                }
            }
        }

        bool requires_grad = left->requires_grad || right->requires_grad;
        auto result = std::make_shared<Tensor>(
            result_data, result_shape, requires_grad,
            std::vector<std::shared_ptr<Tensor>>{left, right}, "matmul"
        );

        result->_backward = [left, right, out = result.get(), num_batches, res_batch, 
                            M, N, K, l_stride_M, l_stride_K, r_stride_K, r_stride_N,
                            left_padded_strides, right_padded_strides]() {
            
            for (std::size_t b = 0; b < num_batches; ++b) {
                std::size_t l_batch_offset = 0;
                std::size_t r_batch_offset = 0;

                if (!res_batch.empty()) {
                    std::vector<std::size_t> batch_multi_idx = Tensor::convert_flat_to_multi_index(b, res_batch);
                    for (std::size_t dim = 0; dim < res_batch.size(); ++dim) {
                        l_batch_offset += batch_multi_idx[dim] * left_padded_strides[dim];
                        r_batch_offset += batch_multi_idx[dim] * right_padded_strides[dim];
                    }
                }

                std::size_t out_batch_offset = b * M * N;

                for (std::size_t i = 0; i < M; ++i) {
                    std::size_t l_row_offset = l_batch_offset + i * l_stride_M;
                    std::size_t out_row_offset = out_batch_offset + i * N;

                    for (std::size_t j = 0; j < N; ++j) {
                        float grad_out = out->grad[out_row_offset + j];
                        if (grad_out == 0.0f) continue;

                        std::size_t r_col_offset = r_batch_offset + j * r_stride_N;

                        for (std::size_t k = 0; k < K; ++k) {
                            std::size_t l_idx = l_row_offset + k * l_stride_K;
                            std::size_t r_idx = r_col_offset + k * r_stride_K;

                            if (left->requires_grad) {
                                left->grad[l_idx] += grad_out * right->data[r_idx];
                            }
                            if (right->requires_grad) {
                                right->grad[r_idx] += left->data[l_idx] * grad_out;
                            }
                        }
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
        return 1 / (1 + std::exp(-x));
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

    std::shared_ptr<Tensor> softmax(const std::shared_ptr<Tensor>& tensor)
    {
        if (tensor->shape.size() != 2)
            throw std::runtime_error("softmax requires a 2D tensor (batch, classes)");

        std::size_t batch = tensor->shape[0];
        std::size_t classes = tensor->shape[1];

        std::vector<float> result_data(tensor->data.size());

        for (std::size_t b{}; b < batch; ++b) {
            std::size_t row_start = b * classes;

            float max_val = tensor->data[row_start];
            for (std::size_t j{1}; j < classes; ++j) {
                max_val = std::max(tensor->data[row_start + j], max_val);
            }
            
            float sum { 0.0f };
            for (std::size_t j{}; j < classes; ++j) {
                float e = std::exp(tensor->data[row_start + j] - max_val);
                result_data[row_start + j] = e;
                sum += e;
            }

            for (std::size_t j{}; j < classes; ++j) {
                result_data[row_start + j] /= sum;
            }
        }

        auto result = std::make_shared<Tensor>(
            result_data,
            tensor->shape,
            tensor->requires_grad,
            std::vector<std::shared_ptr<Tensor>>{tensor},
            "softmax"
        );

        result->_backward = [tensor, out = result.get(), batch, classes]() {
            if (tensor->requires_grad) {
                for (std::size_t b{}; b < batch; ++b) {
                    std::size_t row_start = b * classes;

                    for (std::size_t j{}; j < classes; ++j) {
                        float grad_j = 0.0f;
                        for (std::size_t i{}; i < classes; ++i) {
                            float p_i = out->data[row_start + i];
                            float p_j = out->data[row_start + j];
                            float delta = (i == j) ? 1.0f : 0.0f;
                            float jacobian_ij = p_i * (delta - p_j);
                            grad_j += out->grad[row_start + i] * jacobian_ij;
                        }
                        tensor->grad[row_start + j] += grad_j;
                    }
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

    std::shared_ptr<Tensor> transpose(const std::shared_ptr<Tensor>& tensor)
    {
        if (tensor->shape.size() != 2) {
            throw std::runtime_error("Transpose can only be called on a 2D tensor");
        }

        std::vector<std::size_t> result_shape = {tensor->shape[1], tensor->shape[0]};
        std::vector<float> result_data(tensor->data.size());
        bool result_requires_grad = tensor->requires_grad;

        for (std::size_t i{}; i < tensor->shape[0]; ++i) {
            for (std::size_t j{}; j < tensor->shape[1]; ++j) {
                std::size_t src_idx = (i * tensor->strides[0]) + (j * tensor->strides[1]);
                std::size_t dest_idx = (j * tensor->shape[0]) + i;
                result_data[dest_idx] = tensor->data[src_idx];
            }
        }

        std::string op = "transpose";

        auto result = std::make_shared<Tensor>(
            result_data, 
            result_shape, 
            result_requires_grad,
            std::vector<std::shared_ptr<Tensor>>{tensor},
            op
        );

        result->_backward = [tensor, out = result.get()]() {
            if (tensor->requires_grad) {
                for (std::size_t i{}; i < tensor->shape[0]; ++i) {
                    for (std::size_t j{}; j < tensor->shape[1]; ++j) {
                        std::size_t src_idx = (i * tensor->strides[0]) + (j * tensor->strides[1]);
                        std::size_t dest_idx = (j * tensor->shape[0]) + i;

                        tensor->grad[src_idx] += out->grad[dest_idx];
                    }
                }
            }
        };

        return result;
    }

    std::shared_ptr<Tensor> batchNorm1d(
            const std::shared_ptr<Tensor>& input,
            const std::shared_ptr<Tensor>& gamma,
            const std::shared_ptr<Tensor>& beta,
            std::shared_ptr<Tensor>& running_mean,
            std::shared_ptr<Tensor>& running_var,
            bool training,
            float momentum,
            float epsilon
        )
    {
        if (input->shape.size() != 2) {
            throw std::runtime_error("Input must be a 2D tensor of shape (batch, classes)");
        }

        std::size_t batch = input->shape[0];
        std::size_t classes = input->shape[1];

        std::vector<float> mean(classes, 0.0f);
        std::vector<float> var(classes, 0.0f);
        std::vector<float> std_dev(classes, 0.0f);
        std::vector<float> result_data(input->data.size());

        // calculating the mean
        for (std::size_t b{}; b < batch; ++b) {
            std::size_t row_start = b * classes;

            for (std::size_t i{}; i < classes; ++i) {
                mean[i] += input->data[row_start + i];
            }
        }

        for (std::size_t i{}; i < classes; ++i) {
            mean[i] /= batch;
        }

        // calculating the variance and the standard deviation
        for (std::size_t b{}; b < batch; ++b) {
            std::size_t row_start = b * classes;

            for (std::size_t i{}; i < classes; ++i) {
                float diff = input->data[row_start + i] - mean[i];
                var[i] += diff * diff;
            }
        }

        for (std::size_t i{}; i < classes; ++i) {
            var[i] /= batch;
            std_dev[i] = std::sqrt(var[i] + epsilon);
        }

        std::vector<float> x_hat(input->data.size());

        // normalizing, scaling&shifting and calculating moving averages
        if (training) {
            for (std::size_t b{}; b < batch; ++b) {
                std::size_t row_start = b * classes;

                for (std::size_t i{}; i < classes; ++i) {
                    x_hat[row_start + i] = (input->data[row_start + i] - mean[i]) / std_dev[i];
                    result_data[row_start + i] = (gamma->data[i] * x_hat[row_start + i]) + beta->data[i];
                }
            }
                
            float bessel_correction = (batch > 1) ? (static_cast<float>(batch) / (batch - 1)) : 1.0f;
            for (std::size_t i{}; i < classes; ++i) {
                running_mean->data[i] = (momentum * mean[i]) + (1.0f - momentum) * running_mean->data[i];
                running_var->data[i] = (momentum * var[i] * bessel_correction) + (1.0f - momentum) * running_var->data[i];
            }
        }
        else {
            for (std::size_t b{}; b < batch; ++b) {
                std::size_t row_start = b * classes;
                
                for (std::size_t i{}; i < classes; ++i) {
                    float r_mean = running_mean->data[i];
                    float r_var = running_var->data[i];
                    
                    x_hat[row_start + i] = (input->data[row_start + i] - r_mean) / std::sqrt(r_var + epsilon);
                    result_data[row_start + i] = (gamma->data[i] * x_hat[row_start + i]) + beta->data[i];
                }
            }
        }

        bool req_grad = input->requires_grad || beta->requires_grad || gamma->requires_grad;

        auto result = std::make_shared<Tensor>(
            result_data,
            input->shape,
            req_grad,
            std::vector<std::shared_ptr<Tensor>>{input, gamma, beta},
            "batchNorm1d"
        );

        result->_backward = [input, out = result.get(), batch, classes, beta, gamma, x_hat, std_dev]() {
            std::vector<float> d_beta(classes, 0.0f);
            std::vector<float> d_gamma(classes, 0.0f);

            for (std::size_t b{}; b < batch; ++b) {
                std::size_t row_start = b * classes;

                for (std::size_t i{}; i < classes; ++i) {
                    d_beta[i] += out->grad[row_start + i]; 
                    d_gamma[i] += out->grad[row_start + i] * x_hat[row_start + i];
                }
            }

            if (beta->requires_grad) {
                for (std::size_t i{}; i < classes; ++i) {
                    beta->grad[i] += d_beta[i];
                }
            }

            if (gamma->requires_grad) {
                for (std::size_t i{}; i < classes; ++i) {
                    gamma->grad[i] += d_gamma[i];
                }
            }

            if (input->requires_grad) {
                for (std::size_t b{}; b < batch; ++b) {
                    std::size_t row_start = b * classes;

                    for (std::size_t i{}; i < classes; ++i) {
                        input->grad[row_start + i] += (gamma->data[i] / (batch * std_dev[i])) * (batch * out->grad[row_start + i] - d_beta[i] - x_hat[row_start + i] * d_gamma[i]);
                    }
                }
            }
        };

        return result;
    }

    std::shared_ptr<Tensor> im2col(
        const std::shared_ptr<Tensor>& img,
        std::size_t kernel_h,
        std::size_t kernel_w,
        std::size_t stride,
        std::size_t padding
    )
    {
        if (img->shape.size() != 4) {
            throw std::invalid_argument("im2col requires an image tensor of 4 dimensions.");
        }

        std::size_t N = img->shape[0];
        std::size_t C_in = img->shape[1];
        std::size_t h_in = img->shape[2];
        std::size_t w_in = img->shape[3];

        std::size_t h_out = ((h_in + 2 * padding - kernel_h) / stride) + 1;
        std::size_t w_out = ((w_in + 2 * padding - kernel_w) / stride) + 1;

        std::size_t rows_out = C_in * kernel_h * kernel_w;
        std::size_t cols_out = N * h_out * w_out;

        std::vector<float> col_data(rows_out * cols_out, 0.0f);

        std::size_t col_idx = 0;
        for (std::size_t n{}; n < N; ++n) {
            for (std::size_t out_h{}; out_h < h_out; ++out_h) {
                for (std::size_t out_w{}; out_w < w_out; ++out_w) {
                    std::size_t row_idx = 0;
                    for (std::size_t c{}; c < C_in; ++c) {
                        for (std::size_t kh{}; kh < kernel_h; ++kh) {
                            for (std::size_t kw{}; kw < kernel_w; ++kw) {
                                int in_h = static_cast<int>(out_h * stride + kh) - static_cast<int>(padding);
                                int in_w = static_cast<int>(out_w * stride + kw) - static_cast<int>(padding);

                                float val = 0.0f;
                                if (in_h >= 0 && in_h < static_cast<int>(h_in) &&
                                    in_w >= 0 && in_w < static_cast<int>(w_in)) {
                                        std::size_t img_idx = n * (C_in * h_in * w_in) + c * (h_in * w_in) + static_cast<std::size_t>(in_h) * w_in + static_cast<std::size_t>(in_w);
                                        val = img->data[img_idx];
                                }

                                col_data[row_idx * cols_out + col_idx] = val;
                                ++row_idx;
                            }
                        }
                    }
                    ++col_idx;
                }
            }
        }

        return std::make_shared<Tensor>(
            col_data,
            std::vector<std::size_t>{rows_out, cols_out},
            img->requires_grad
        );
    }

    std::shared_ptr<Tensor> col2im(
        const std::shared_ptr<Tensor>& dX_col,
        const std::vector<std::size_t>& img_shape,
        std::size_t kernel_h,
        std::size_t kernel_w,
        std::size_t stride,
        std::size_t padding
    )
    {
        std::size_t N = img_shape[0];
        std::size_t C_in = img_shape[1];
        std::size_t h_in = img_shape[2];
        std::size_t w_in = img_shape[3];

        std::size_t h_out = ((h_in + 2 * padding - kernel_h) / stride) + 1;
        std::size_t w_out = ((w_in + 2 * padding - kernel_w) / stride) + 1;

        std::size_t channels_col = C_in * kernel_h * kernel_w;
        std::size_t num_cols = N * h_out * w_out;

        std::vector<float> dX_data(N * C_in * h_in * w_in, 0.0f);

        std::size_t col_idx = 0;
        for (std::size_t n = 0; n < N; ++n) {
            for (std::size_t out_h = 0; out_h < h_out; ++out_h) {
                for (std::size_t out_w = 0; out_w < w_out; ++out_w) {
                    std::size_t row_idx = 0;
                    for (std::size_t c = 0; c < C_in; ++c) {
                        for (std::size_t kh = 0; kh < kernel_h; ++kh) {
                            for (std::size_t kw = 0; kw < kernel_w; ++kw) {
                                
                                int in_h = static_cast<int>(out_h * stride + kh) - static_cast<int>(padding);
                                int in_w = static_cast<int>(out_w * stride + kw) - static_cast<int>(padding);

                                if (in_h >= 0 && in_h < static_cast<int>(h_in) &&
                                    in_w >= 0 && in_w < static_cast<int>(w_in)) {
                                    
                                    std::size_t img_idx = n * (C_in * h_in * w_in) + 
                                                        c * (h_in * w_in) + 
                                                        static_cast<std::size_t>(in_h) * w_in + 
                                                        static_cast<std::size_t>(in_w);

                                    float grad_val = dX_col->data[row_idx * num_cols + col_idx];
                                    
                                    dX_data[img_idx] += grad_val;
                                }
                                row_idx++;
                            }
                        }
                    }
                    col_idx++;
                }
            }
        }

        return std::make_shared<Tensor>(dX_data, img_shape);
    }

    std::shared_ptr<Tensor> conv2d(
        const std::shared_ptr<Tensor>& input,
        const std::shared_ptr<Tensor>& weight,
        const std::shared_ptr<Tensor>& bias,
        std::size_t stride,
        std::size_t padding
    )
    {
        if (input->shape.size() != 4) {
            throw std::invalid_argument("conv2d requires an input tensor of 4 dimensions.");
        }

        if (weight->shape.size() != 4) {
            throw std::invalid_argument("conv2d requires a kernel tensor of 4 dimensions.");
        }

        if (input->shape[1] != weight->shape[1]) {
            throw std::invalid_argument("Input channels must match weight input channels.");
        }

        std::size_t N = input->shape[0];
        std::size_t C_out = weight->shape[0];
        std::size_t C_in = weight->shape[1];
        std::size_t kernel_h = weight->shape[2];
        std::size_t kernel_w = weight->shape[3];

        std::size_t h_out = ((input->shape[2] + 2 * padding - kernel_h) / stride) + 1;
        std::size_t w_out = ((input->shape[3] + 2 * padding - kernel_w) / stride) + 1;

        auto X_col = im2col(input, kernel_h, kernel_w, stride, padding);
        
        std::size_t K_flat = C_in * kernel_h * kernel_w;
        auto W_col = std::make_shared<Tensor>(
            weight->data,
            std::vector<std::size_t>{C_out, K_flat},
            weight->requires_grad
        );

        auto out_col = matmul(W_col, X_col);

        std::vector<float> result_data(N * C_out * h_out * w_out);

        for (std::size_t n{}; n < N; ++n) {
            for (std::size_t c{}; c < C_out; ++c) {
                float b_val = (bias != nullptr) ? bias->data[c] : 0.0f;
                for (std::size_t h{}; h < h_out; ++h) {
                    for (std::size_t w{}; w < w_out; ++w) {
                        std::size_t col_idx = n * (h_out * w_out) + h * w_out + w;
                        std::size_t out_col_idx = c * (N * h_out * w_out) + col_idx;
                        std::size_t final_idx = n * (C_out * h_out * w_out) + c * (h_out * w_out) + h * w_out + w;
                        result_data[final_idx] = out_col->data[out_col_idx] + b_val;
                    }
                }
            }
        }

        bool req_grad = input->requires_grad || weight->requires_grad || (bias && bias->requires_grad);
        std::vector<std::shared_ptr<Tensor>> parents{input, weight};
        if (bias) parents.push_back(bias);

        auto result = std::make_shared<Tensor>(
            result_data,
            std::vector<std::size_t>{N, C_out, h_out, w_out},
            req_grad,
            parents,
            "conv2d"
        );

        result->_backward = [input, weight, bias, out = result.get(), X_col, W_col, stride, padding, N, C_out, C_in, kernel_h, kernel_w, h_out, w_out]() {
            std::size_t num_cols = N * h_out * w_out;
            std::size_t K_flat = C_in * kernel_h * kernel_w;

            std::vector<float> dOut_col_data(C_out * num_cols);
            for (std::size_t n{}; n < N; ++n) {
                for (std::size_t c{}; c < C_out; ++c) {
                    for (std::size_t h{}; h < h_out; ++h) {
                        for (std::size_t w{}; w < w_out; ++w) {
                            std::size_t final_idx = n * (C_out * h_out * w_out) + c * (h_out * w_out) + h * w_out + w;
                            std::size_t col_idx = n * (h_out * w_out) + h * w_out + w;
                            std::size_t out_col_idx = c * num_cols + col_idx;
                            dOut_col_data[out_col_idx] = out->grad[final_idx];
                        }
                    }
                }
            }

            auto dOut_col = std::make_shared<Tensor>(dOut_col_data, std::vector<std::size_t>{C_out, num_cols});

            if (bias && bias->requires_grad) {
                for (std::size_t c = 0; c < C_out; ++c) {
                    float sum = 0.0f;
                    for (std::size_t col = 0; col < num_cols; ++col) {
                        sum += dOut_col->data[c * num_cols + col];
                    }
                    bias->grad[c] += sum;
                }
            }

            if (weight->requires_grad) {
                auto X_col_T = transpose(X_col);
                auto dW_col = matmul(dOut_col, X_col_T);

                for (std::size_t i = 0; i < weight->data.size(); ++i) {
                    weight->grad[i] += dW_col->data[i];
                }
            }

            if (input->requires_grad) {
                auto W_col_T = transpose(W_col);
                auto dX_col = matmul(W_col_T, dOut_col);

                auto dX = col2im(dX_col, input->shape, kernel_h, kernel_w, stride, padding);

                for (std::size_t i = 0; i < input->data.size(); ++i) {
                    input->grad[i] += dX->data[i];
                }
            }
        };

        return result;
    }

    std::shared_ptr<Tensor> maxPool2d(
        const std::shared_ptr<Tensor>& input,
        std::size_t kernel_h,
        std::size_t kernel_w,
        std::size_t stride,
        std::size_t padding
    )
    {
        if (input->shape.size() != 4) {
            throw std::invalid_argument("maxPool2d requires an input tensor of 4 dimensions.");
        }

        if (padding >= kernel_h || padding >= kernel_w) {
            throw std::invalid_argument("maxPool2d: padding must be smaller than kernel size.");
        }

        std::size_t N = input->shape[0];
        std::size_t C = input->shape[1];
        std::size_t H_in = input->shape[2];
        std::size_t W_in = input->shape[3];

        std::size_t H_out = ((H_in + 2 * padding - kernel_h) / stride) + 1;
        std::size_t W_out = ((W_in + 2 * padding - kernel_w) / stride) + 1;

        std::vector<float> result_data(N * C * H_out * W_out, 0.0f);

        auto argmax_cache = std::make_shared<std::vector<std::size_t>>(N * C * H_out * W_out, 0);

        for (std::size_t n{}; n < N; ++n) {
            for (std::size_t c{}; c < C; ++c) {
                for (std::size_t oh{}; oh < H_out; ++oh) {
                    for (std::size_t ow{}; ow < W_out; ++ow) {
                        float max_val = -std::numeric_limits<float>::infinity();
                        std::size_t max_idx = 0;
                        bool found = false;

                        for (std::size_t kh{}; kh < kernel_h; ++kh) {
                            for (std::size_t kw{}; kw < kernel_w; ++kw) {
                                int ih = static_cast<int>(oh * stride + kh) - static_cast<int>(padding);
                                int iw = static_cast<int>(ow * stride + kw) - static_cast<int>(padding);

                                if (ih >= 0 && ih < static_cast<int>(H_in) &&
                                    iw >= 0 && iw < static_cast<int>(W_in)) {

                                    std::size_t in_idx = n * (C * H_in * W_in) + c * (H_in * W_in) + static_cast<std::size_t>(ih) * W_in + static_cast<std::size_t>(iw);

                                    float val = input->data[in_idx];
                                    if (!found || val > max_val) {
                                        max_val = val;
                                        max_idx = in_idx;
                                        found = true;
                                    }
                                }
                            }
                        }

                        std::size_t out_idx = n * (C * H_out * W_out) + c * (H_out * W_out) + oh * W_out + ow;
                        result_data[out_idx] = max_val;
                        (*argmax_cache)[out_idx] = max_idx;   
                    }
                }
            }
        }

        auto result = std::make_shared<Tensor>(
            result_data,
            std::vector<std::size_t>({N, C, H_out, W_out}),
            input->requires_grad,
            std::vector<std::shared_ptr<Tensor>>{input},
            "maxpool2d"
        );

        result->_backward = [input, out = result.get(), argmax_cache]() {
            if (input->requires_grad) {
                for (std::size_t i{}; i < out->grad.size(); ++i) {
                    std::size_t winning_idx = (*argmax_cache)[i];
                    input->grad[winning_idx] += out->grad[i];
                }
            }
        };

        return result;
    }

    std::shared_ptr<Tensor> flatten(std::shared_ptr<Tensor>& input)
    {
        if (input->shape.size() < 3) {
            throw std::invalid_argument("cannot flatten a tensor with dimensions < 3.");
        }

        std::vector<std::size_t> result_shape{input->shape[0], 1};
        for (std::size_t i{1}; i < input->shape.size(); ++i) {
            result_shape[1] *= input->shape[i];
        }

        auto result = std::make_shared<Tensor>(
            input->data,
            result_shape,
            input->requires_grad,
            std::vector<std::shared_ptr<Tensor>>{input},
            "flatten"
        );

        result->_backward = [input, out = result.get()]() {
            if (input->requires_grad) {
                std::transform(
                    input->grad.begin(),
                    input->grad.end(),
                    out->grad.begin(),
                    input->grad.begin(),
                    std::plus<float>()
                );
            }
        };

        return result;
    }

    std::shared_ptr<Tensor> max(const std::shared_ptr<Tensor>& tensor)
    {
        auto max_it = std::max_element(tensor->data.begin(), tensor->data.end());
        std::size_t max_idx = std::distance(tensor->data.begin(), max_it);

        std::vector<std::size_t> result_shape = {};
        std::vector<float> result_data = {tensor->data[max_idx]};
        bool result_requires_grad = tensor->requires_grad;
        std::string op = "max";

        auto result = std::make_shared<Tensor>(
            result_data, 
            result_shape, 
            result_requires_grad,
            std::vector<std::shared_ptr<Tensor>>{tensor},
            op
        );

        result->_backward = [tensor, out = result.get(), max_idx]() {
            if (tensor->requires_grad) {
                tensor->grad[max_idx] += out->grad[0];
            }
        };

        return result;
    }

    std::shared_ptr<Tensor> log(const std::shared_ptr<Tensor>& tensor)
    {
        // to handle cases when x=0
        float epsilon = 1e-8;

        std::vector<float> result_data(tensor->data.size());
        
        for (std::size_t i{}; i < result_data.size(); ++i) {
            result_data[i] = std::log(tensor->data[i] + epsilon);
        }

        std::string op = "log";

        auto result = std::make_shared<Tensor>(
            result_data, 
            tensor->shape, 
            tensor->requires_grad,
            std::vector<std::shared_ptr<Tensor>>{tensor},
            op
        );

        result->_backward = [tensor, out = result.get(), epsilon]() {
            if (tensor->requires_grad) {
                for (std::size_t i{}; i < tensor->grad.size(); ++i) {
                    tensor->grad[i] += (1.0f / (tensor->data[i] + epsilon)) * out->grad[i];
                }
            }
        };

        return result;
    }

    std::shared_ptr<Tensor> exp(const std::shared_ptr<Tensor>& tensor)
    {
        std::vector<float> result_data(tensor->data.size());
        
        for (std::size_t i{}; i < result_data.size(); ++i) {
            result_data[i] = std::exp(tensor->data[i]);
        }

        std::string op = "exp";

        auto result = std::make_shared<Tensor>(
            result_data, 
            tensor->shape, 
            tensor->requires_grad,
            std::vector<std::shared_ptr<Tensor>>{tensor},
            op
        );

        result->_backward = [tensor, out = result.get()]() {
            if (tensor->requires_grad) {
                for (std::size_t i{}; i < tensor->grad.size(); ++i) {
                    tensor->grad[i] += out->data[i] * out->grad[i];
                }
            }
        };

        return result;
    }

    void Tensor::build_topo(
            const std::shared_ptr<Tensor>& node,
            std::vector<std::shared_ptr<Tensor>>& topo,
            std::unordered_set<std::shared_ptr<Tensor>>& visited
        )
    {
        if (visited.contains(node))
            return;

        visited.insert(node);
        for (const std::shared_ptr<Tensor>& child : node->_children) {
            build_topo(child, topo, visited);
        }
        topo.push_back(node);
    }

    void Tensor::backward()
    {
        if (this->data.size() != 1) {
            throw std::runtime_error("backward can only be called on scalar tensors");
        }

        std::vector<std::shared_ptr<Tensor>> topo;
        std::unordered_set<std::shared_ptr<Tensor>> visited;

        build_topo(shared_from_this(), topo, visited);

        if (this->requires_grad)
            this->grad[0] = 1.0f;

        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->_backward();
        }
    }

    std::shared_ptr<Tensor> mse_loss(const std::shared_ptr<Tensor>& prediction, const std::shared_ptr<Tensor>& target)
    {
        auto diff = prediction - target;
        auto squared = diff * diff;
        auto mse = mean(squared);

        return mse;
    }

    std::shared_ptr<Tensor> ce_loss(const std::shared_ptr<Tensor>& logits, const std::vector<std::size_t>& target_classes)
    {
        if (logits->shape.size() != 2)
        throw std::runtime_error("ce_loss requires logits of shape (batch, num_classes)");

        std::size_t batch = logits->shape[0];
        std::size_t classes = logits->shape[1];

        if (target_classes.size() != batch)
            throw std::runtime_error("target_classes size must match batch size");

        std::vector<float> softmax_data(batch * classes);
        float total_loss = 0.0f;

        for (std::size_t i{}; i < batch; ++i) {
            std::size_t row_start = i * logits->strides[0];

            // log-sum-exp trick, per row
            float max_val = logits->data[row_start];
            for (std::size_t j = 1; j < classes; ++j) {
                float val = logits->data[row_start + j * logits->strides[1]];
                if (val > max_val) max_val = val;
            }

            float sum_exp = 0.0f;
            for (std::size_t j{}; j < classes; ++j) {
                float val = logits->data[row_start + j * logits->strides[1]];
                sum_exp += std::exp(val - max_val);
            }
            float log_sum_exp = max_val + std::log(sum_exp);

            float x_target = logits->data[row_start + target_classes[i] * logits->strides[1]];
            total_loss += (log_sum_exp - x_target);

            for (std::size_t j{}; j < classes; ++j) {
                float val = logits->data[row_start + j * logits->strides[1]];
                softmax_data[i * classes + j] = std::exp(val - log_sum_exp);
            }
        }

        float mean_loss = total_loss / static_cast<float>(batch);

        auto result = std::make_shared<Tensor>(
            std::vector<float>{mean_loss},
            std::vector<std::size_t>{},
            logits->requires_grad,
            std::vector<std::shared_ptr<Tensor>>{logits},
            "cross_entropy"
        );

        result->_backward = [logits, out = result.get(), softmax_data, target_classes, batch, classes]() {
            if (logits->requires_grad) {
                for (std::size_t i{}; i < batch; ++i) {
                    std::size_t row_start = i * logits->strides[0];
                    for (std::size_t j{}; j < classes; ++j) {
                        float grad_val = softmax_data[i * classes + j];
                        if (j == target_classes[i]) {
                            grad_val -= 1.0f;
                        }
                        logits->grad[row_start + j * logits->strides[1]] +=
                            (grad_val / static_cast<float>(batch)) * out->grad[0];
                    }
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