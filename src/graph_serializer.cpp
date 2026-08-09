#include <zerograd/graph_serializer.hpp>
#include <fstream>
#include <sstream>

namespace zerograd
{
    void GraphSerializer::export_graph_to_json(
        const std::unordered_map<std::shared_ptr<Tensor>, std::tuple<std::size_t, std::size_t, std::size_t>>& intervals,
        std::string filename
    )
    {

    }
}