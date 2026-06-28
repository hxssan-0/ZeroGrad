#include <zerograd/scalar.h>
#include <cmath>
#include <algorithm>

namespace zerograd
{
    Scalar::Scalar(
        float data,
        std::vector<std::shared_ptr<Scalar>> children,
        std::string op
    )
    : data(data), _children(std::move(children)), _op(std::move(op))
    {
    }

    const std::string& Scalar::get_op() const
    {
        return _op;
    }

    const std::vector<std::shared_ptr<Scalar>>& Scalar::get_children() const
    {
        return _children;
    }

    void Scalar::call_backward()
    {
        _backward();
    }
    
    std::shared_ptr<Scalar> operator+(const std::shared_ptr<Scalar>& left, const std::shared_ptr<Scalar>& right)
    {
        auto result = std::make_shared<Scalar>(
            left->data + right->data, 
            std::vector<std::shared_ptr<Scalar>>{left, right}, 
            "+"
        );
        
        result->_backward = [left, right, out = result.get()]() {
            left->grad += 1.0f * out->grad;
            right->grad += 1.0f * out->grad;
        };

        return result;
    }

    std::shared_ptr<Scalar> operator*(const std::shared_ptr<Scalar>& left, const std::shared_ptr<Scalar>& right)
    {
        auto result = std::make_shared<Scalar>(
            left->data * right->data, 
            std::vector<std::shared_ptr<Scalar>>{left, right}, 
            "*"
        );
        
        result->_backward = [left, right, out = result.get()]() {
            left->grad += right->data * out->grad;
            right->grad += left->data * out->grad;
        };

        return result;
    }

    std::shared_ptr<Scalar> operator+(const std::shared_ptr<Scalar>& left, float right)
    {
        auto right_obj = std::make_shared<Scalar>(right);
        return left + right_obj;
    }

    std::shared_ptr<Scalar> operator+(float left, const std::shared_ptr<Scalar>& right)
    {
        auto left_obj = std::make_shared<Scalar>(left);
        return left_obj + right;
    }

    std::shared_ptr<Scalar> operator*(const std::shared_ptr<Scalar>& left, float right)
    {
        auto right_obj = std::make_shared<Scalar>(right);
        return left * right_obj;
    }

    std::shared_ptr<Scalar> operator*(float left, const std::shared_ptr<Scalar>& right)
    {
        auto left_obj = std::make_shared<Scalar>(left);
        return left_obj * right;
    }

    std::shared_ptr<Scalar> operator-(const std::shared_ptr<Scalar>& scalar)
    {
        return scalar * -1.0f;
    }

    std::shared_ptr<Scalar> operator-(const std::shared_ptr<Scalar>& left, const std::shared_ptr<Scalar>& right)
    {
        return left + (-right);
    }

    std::shared_ptr<Scalar> operator-(const std::shared_ptr<Scalar>& left, float right)
    {
        return left + (-right);
    }

    std::shared_ptr<Scalar> operator-(float left, const std::shared_ptr<Scalar>& right)
    {
        return left + (-right);
    }

    std::shared_ptr<Scalar> pow(const std::shared_ptr<Scalar>& scalar, float exponent)
    {
        auto result = std::make_shared<Scalar>(
            std::pow(scalar->data, exponent),
            std::vector<std::shared_ptr<Scalar>>{scalar},
            "^" + std::to_string(exponent)
        );

        result->_backward = [scalar, exponent, out = result.get()]() {
            scalar->grad += exponent * (std::pow(scalar->data, exponent - 1)) * out->grad;
        };

        return result;
    }

    std::shared_ptr<Scalar> operator/(const std::shared_ptr<Scalar>& left, const std::shared_ptr<Scalar>& right)
    {
        return left * pow(right, -1.0f);
    }

    std::shared_ptr<Scalar> operator/(const std::shared_ptr<Scalar>& left, float right)
    {
        return left * (1.0f / right);
    }

    std::shared_ptr<Scalar> operator/(float left, const std::shared_ptr<Scalar>& right)
    {
        return left * pow(right, -1.0f);
    }

    std::shared_ptr<Scalar> exp(const std::shared_ptr<Scalar>& scalar)
    {
        auto result = std::make_shared<Scalar>(
            std::exp(scalar->data), 
            std::vector<std::shared_ptr<Scalar>>{scalar},
            "exp"
        );

        result->_backward = [scalar, out = result.get()]() {
            scalar->grad += out->data * out->grad;
        };

        return result;
    }

    std::shared_ptr<Scalar> tanh(const std::shared_ptr<Scalar>& scalar)
    {
        float t = std::tanh(scalar->data);
        auto result = std::make_shared<Scalar>(t, std::vector<std::shared_ptr<Scalar>>{scalar}, "tanh");

        result->_backward = [scalar, out = result.get()]() {
            scalar->grad += (1.0f - (out->data * out->data)) * out->grad;
        };

        return result;
    }

    std::shared_ptr<Scalar> relu(const std::shared_ptr<Scalar>& scalar)
    {
        auto result = std::make_shared<Scalar>(
            std::max(0.0f, scalar->data),
            std::vector<std::shared_ptr<Scalar>>{scalar},
            "relu"
        );

        result->_backward = [scalar, out = result.get()]() {
            scalar->grad += ((out->data == 0) ? 0 : 1) * out->grad;
        };

        return result;
    }

    void Scalar::build_topo(
            const std::shared_ptr<Scalar>& node,
            std::vector<std::shared_ptr<Scalar>>& topo,
            std::unordered_set<std::shared_ptr<Scalar>>& visited
        )
    {
        if (visited.contains(node))
            return;

        visited.insert(node);
        for (const std::shared_ptr<Scalar>& child : node->_children) {
            build_topo(child, topo, visited);
        }
        topo.push_back(node);
    }

    void Scalar::backward()
    {
        std::vector<std::shared_ptr<Scalar>> topo;
        std::unordered_set<std::shared_ptr<Scalar>> visited;

        build_topo(shared_from_this(), topo, visited);

        this->grad = 1.0f;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->_backward();
        }
    }

    std::ostream& operator<<(std::ostream& out, const std::shared_ptr<Scalar>& scalar)
    {
        if (!scalar)
            return out;

        out << "Scalar ( data = " << scalar->data << " | grad = " << scalar->grad << " | op = " 
            << (scalar->_op == "" ? "none" : scalar->_op) << " )";

        return out;
    }
}