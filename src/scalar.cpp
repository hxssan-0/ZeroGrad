#include <zerograd/scalar.h>

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

    std::ostream& operator<<(std::ostream& out, const std::shared_ptr<Scalar>& scalar)
    {
        if (!scalar)
            return out;

        out << "Scalar ( data = " << scalar->data << " | grad = " << scalar->grad << " | op = " 
            << (scalar->_op == "" ? "none" : scalar->_op) << " )";

        return out;
    }
}