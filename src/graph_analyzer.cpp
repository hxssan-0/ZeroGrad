#include <zerograd/graph_analyzer.hpp>

namespace zerograd
{
    std::unordered_map<std::shared_ptr<Tensor>, std::tuple<std::size_t, std::size_t, std::size_t>> GraphAnalyzer::dry_forward(
        const std::shared_ptr<Tensor>& node
    )
    {
        std::unordered_map<std::shared_ptr<Tensor>, std::tuple<std::size_t, std::size_t, std::size_t>> result;

        std::vector<std::shared_ptr<Tensor>> topo = Tensor::get_topo_order(node);

        std::unordered_map<std::shared_ptr<Tensor>, std::size_t> remaining_refs;
        std::size_t max_birth = 0;
        for (auto& t : topo) {
            remaining_refs[t] = t->ref_count;
            if (t->birth_step > max_birth) {
                max_birth = t->birth_step;
            }
        }

        std::size_t sim_step = max_birth + 1;
        std::size_t max_possible_death = sim_step + topo.size();
        for (auto& t : topo) {
            std::size_t data_grad_bytes = t->data.size() * sizeof(float) * 2;
            
            std::size_t shape_bytes = t->shape.size() * sizeof(std::size_t);
            std::size_t stride_bytes = t->strides.size() * sizeof(std::size_t);
            
            std::size_t total_bytes = data_grad_bytes + shape_bytes + stride_bytes;
            result[t] = {t->birth_step, max_possible_death, total_bytes};
        }

        bool is_root = true;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            if (is_root) {
                std::get<1>(result[*it]) = sim_step;
                is_root = false;
            }
            
            for (auto& child : (*it)->_children) {
                if (--remaining_refs[child] == 0) {
                    std::get<1>(result[child]) = sim_step;
                }
            }

            ++sim_step;
        }

        return result;
    }
}