#include <zerograd/memory_planner.hpp>
#include <vector>
#include <algorithm>
#include <ranges>

namespace zerograd
{
    std::pair<std::size_t, std::unordered_map<std::shared_ptr<Tensor>, std::size_t>> MemoryPlanner::plan_memory(
        const std::unordered_map<std::shared_ptr<Tensor>, 
        std::tuple<std::size_t, std::size_t, std::size_t>>& liveness_info
    )
    {
        // sorting the tensors by size (descending)
        using LivenessPair = std::pair<std::shared_ptr<Tensor>, std::tuple<std::size_t, std::size_t, std::size_t>>;
        std::vector<LivenessPair> sorted_tensors(liveness_info.begin(), liveness_info.end());

        std::ranges::sort(sorted_tensors, [](const LivenessPair& a, const LivenessPair& b) {
            std::size_t size_a = std::get<2>(a.second);
            std::size_t size_b = std::get<2>(b.second);
            if (size_a != size_b)
                return size_a > size_b;
            return std::get<0>(a.second) < std::get<0>(b.second); // using birth step as a tie-breaker
        });

        // applying the greedy first fit algo
        std::unordered_map<std::shared_ptr<Tensor>, std::size_t> tensor_offsets;
        std::size_t total_size = 0;

        for (auto& [tA, tA_info] : sorted_tensors) {
            std::size_t birth_A = std::get<0>(tA_info);
            std::size_t death_A = std::get<1>(tA_info);
            std::size_t size_A = std::get<2>(tA_info);

            std::vector<std::pair<std::size_t, std::size_t>> occupied; // (start, end) of conflicting pairs of tensors

            for (auto& [tB, offset_B] : tensor_offsets) {
                auto& tB_info = liveness_info.at(tB);
                std::size_t birth_B = std::get<0>(tB_info);
                std::size_t death_B = std::get<1>(tB_info);
                std::size_t size_B = std::get<2>(tB_info);

                bool time_overlap = (birth_A < death_B) && (birth_B < death_A);
                if (time_overlap) {
                    occupied.emplace_back(offset_B, offset_B + size_B);
                }
            }

            std::ranges::sort(occupied);
            
            std::size_t offset_A = 0;
            for (auto& [occupied_start, occupied_end] : occupied) {
                if (offset_A + size_A <= occupied_start)
                    break;

                offset_A = std::max(offset_A, occupied_end);
            }

            tensor_offsets[tA] = offset_A;
            total_size = std::max(total_size, offset_A + size_A);
        }

        return {total_size, tensor_offsets};
    }
}