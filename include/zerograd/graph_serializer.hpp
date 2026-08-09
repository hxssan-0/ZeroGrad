#pragma once

#include "tensor.hpp"
#include <unordered_map>
#include <string>
#include <tuple>

namespace zerograd
{
    class GraphSerializer
    {
        void export_graph_to_json(
            const std::unordered_map<std::shared_ptr<Tensor>, std::tuple<std::size_t, std::size_t, std::size_t>>& intervals,
            std::string filename = "intervals.json"
        );
    };
}