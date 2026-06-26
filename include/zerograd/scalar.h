#pragma once

#include <vector>
#include <string>
#include <functional>
#include <memory>
#include <utility>

namespace zerograd {

    class Scalar {
    private:
        std::vector<std::shared_ptr<Scalar>> _children;
        std::string _op;
        std::function<void()> _backward = [](){};

    public:
        float data{};
        float grad{};

        explicit Scalar(float data,
                        std::vector<std::shared_ptr<Scalar>> children = {},
                        std::string op = "")
            : data(data), _children(std::move(children)), _op(std::move(op)) {}
    };
}