#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include <utility>

namespace zerograd
{
    class Scalar 
    {
    private:
        std::vector<std::shared_ptr<Scalar>> _children;
        std::string _op;
        std::function<void()> _backward = [](){};
    public:
        float data{};
        float grad{};

        explicit Scalar(
            float data,
            std::vector<std::shared_ptr<Scalar>> children = {},
            std::string op = ""
        );

        const std::string& get_op() const;
        const std::vector<std::shared_ptr<Scalar>>& get_children() const;
        void call_backward(); // to be able to call _backward in some test cases since it is private

        friend std::shared_ptr<Scalar> operator+(const std::shared_ptr<Scalar>& left, const std::shared_ptr<Scalar>& right);
        friend std::shared_ptr<Scalar> operator+(const std::shared_ptr<Scalar>& left, float right);
        friend std::shared_ptr<Scalar> operator+(float left, const std::shared_ptr<Scalar>& right);
        
        friend std::shared_ptr<Scalar> operator*(const std::shared_ptr<Scalar>& left, const std::shared_ptr<Scalar>& right);
        friend std::shared_ptr<Scalar> operator*(const std::shared_ptr<Scalar>& left, float right);
        friend std::shared_ptr<Scalar> operator*(float left, const std::shared_ptr<Scalar>& right);

        friend std::shared_ptr<Scalar> operator-(const std::shared_ptr<Scalar>& scalar); // unary negation op
        friend std::shared_ptr<Scalar> operator-(const std::shared_ptr<Scalar>& left, const std::shared_ptr<Scalar>& right);
        friend std::shared_ptr<Scalar> operator-(const std::shared_ptr<Scalar>& left, float right);
        friend std::shared_ptr<Scalar> operator-(float left, const std::shared_ptr<Scalar>& right);

        friend std::shared_ptr<Scalar> pow(const std::shared_ptr<Scalar>& scalar, float exponent);

        friend std::shared_ptr<Scalar> operator/(const std::shared_ptr<Scalar>& left, const std::shared_ptr<Scalar>& right);
        friend std::shared_ptr<Scalar> operator/(const std::shared_ptr<Scalar>& left, float right);
        friend std::shared_ptr<Scalar> operator/(float left, const std::shared_ptr<Scalar>& right);

        friend std::shared_ptr<Scalar> exp(const std::shared_ptr<Scalar>& scalar);

        // activation functions
        friend std::shared_ptr<Scalar> tanh(const std::shared_ptr<Scalar>& scalar);
        friend std::shared_ptr<Scalar> relu(const std::shared_ptr<Scalar>& scalar);

        friend std::ostream& operator<<(std::ostream& out, const std::shared_ptr<Scalar>& scalar);
    };
}