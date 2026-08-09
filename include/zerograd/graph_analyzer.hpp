#include "tensor.hpp"
#include <tuple>
#include <map>
#include <unordered_map>

namespace zerograd
{
    class GraphAnalyzer
    {
        public:
        std::unordered_map<std::shared_ptr<Tensor>, std::tuple<std::size_t, std::size_t, std::size_t>> dry_forward(
            const std::shared_ptr<Tensor>& node
        );
    };
}