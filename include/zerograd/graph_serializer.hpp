#pragma once

#include "tensor.hpp"
#include <unordered_set>
#include <string>
#include <memory>

namespace zerograd
{
    class GraphSerializer
    {  
    private:
        static bool has_cycle_dfs(
            const std::shared_ptr<Tensor>& node,
            std::unordered_set<std::shared_ptr<Tensor>>& gray,
            std::unordered_set<std::shared_ptr<Tensor>>& black
        );

        static std::string escape_json_string(const std::string& s);
    public:
        static void serialize(const std::shared_ptr<Tensor>& root, const std::string& output_path);

        static bool validate_dag(const std::shared_ptr<Tensor>& root);
    };
}