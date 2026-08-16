#pragma once

#include "tensor.hpp"
#include <utility>
#include <unordered_map>
#include <cstddef>
#include <memory>
#include <tuple>

namespace zerograd
{
    class MemoryPlanner
    {
        public:
        static std::pair<std::size_t, std::unordered_map<std::shared_ptr<Tensor>, std::size_t>> plan_memory(
            const std::unordered_map<std::shared_ptr<Tensor>, 
            std::tuple<std::size_t, std::size_t, std::size_t>>& liveness_info
        );
    };
}