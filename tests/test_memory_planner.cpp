#include <zerograd/memory_planner.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <zerograd/tensor.hpp>
#include <algorithm>
#include <limits>
#include <vector>

static std::shared_ptr<zerograd::Tensor> dummy_tensor() {
    return std::make_shared<zerograd::Tensor>(
        std::vector<float>{0.0f}, std::vector<std::size_t>{1}, false);
}

static void assert_no_conflicts(
    const std::unordered_map<std::shared_ptr<zerograd::Tensor>, std::tuple<std::size_t,std::size_t,std::size_t>>& liveness,
    const std::unordered_map<std::shared_ptr<zerograd::Tensor>, std::size_t>& offsets)
{
    for (auto& [tA, infoA] : liveness) {
        for (auto& [tB, infoB] : liveness) {
            if (tA == tB) continue;
            auto [birthA, deathA, sizeA] = infoA;
            auto [birthB, deathB, sizeB] = infoB;
            bool time_overlap = (birthA < deathB) && (birthB < deathA);
            if (!time_overlap) continue;

            std::size_t offA = offsets.at(tA);
            std::size_t offB = offsets.at(tB);
            bool mem_overlap = (offA < offB + sizeB) && (offB < offA + sizeA);
            REQUIRE_FALSE(mem_overlap);
        }
    }
}

static std::pair<std::size_t, std::unordered_map<std::shared_ptr<zerograd::Tensor>, std::size_t>>
solve_optimal(const std::unordered_map<std::shared_ptr<zerograd::Tensor>, std::tuple<std::size_t, std::size_t, std::size_t>>& liveness)
{
    using TensorPtr = std::shared_ptr<zerograd::Tensor>;
    std::vector<TensorPtr> tensors;
    tensors.reserve(liveness.size());
    for (const auto& [t, info] : liveness) {
        tensors.push_back(t);
    }

    std::sort(tensors.begin(), tensors.end());

    std::size_t best_total = std::numeric_limits<std::size_t>::max();
    std::unordered_map<TensorPtr, std::size_t> best_offsets;

    do {
        std::unordered_map<TensorPtr, std::size_t> current_offsets;
        std::size_t current_total = 0;

        for (const auto& tA : tensors) {
            auto [birthA, deathA, sizeA] = liveness.at(tA);

            std::vector<std::pair<std::size_t, std::size_t>> occupied;
            for (const auto& [tB, offB] : current_offsets) {
                auto [birthB, deathB, sizeB] = liveness.at(tB);
                bool time_overlap = (birthA < deathB) && (birthB < deathA);
                if (time_overlap) {
                    occupied.emplace_back(offB, offB + sizeB);
                }
            }

            std::sort(occupied.begin(), occupied.end());

            std::size_t offA = 0;
            for (const auto& [start, end] : occupied) {
                if (offA + sizeA <= start) break;
                offA = std::max(offA, end);
            }

            current_offsets[tA] = offA;
            current_total = std::max(current_total, offA + sizeA);
        }

        if (current_total < best_total) {
            best_total = current_total;
            best_offsets = current_offsets;
        }

    } while (std::next_permutation(tensors.begin(), tensors.end()));

    return {best_total, best_offsets};
}

TEST_CASE("plan_memory: single tensor gets offset 0") {
    auto t = dummy_tensor();
    std::unordered_map<std::shared_ptr<zerograd::Tensor>, std::tuple<std::size_t,std::size_t,std::size_t>> liveness{
        {t, {0, 5, 100}}
    };

    auto [total, offsets] = zerograd::MemoryPlanner::plan_memory(liveness);

    REQUIRE(offsets.at(t) == 0);
    REQUIRE(total == 100);
}

TEST_CASE("plan_memory: non-overlapping lifetimes fully reuse the same memory") {
    auto a = dummy_tensor();
    auto b = dummy_tensor();
    auto c = dummy_tensor();

    std::unordered_map<std::shared_ptr<zerograd::Tensor>, std::tuple<std::size_t,std::size_t,std::size_t>> liveness{
        {a, {0, 2, 100}},
        {b, {2, 4, 100}},
        {c, {4, 6, 100}},
    };

    auto [total, offsets] = zerograd::MemoryPlanner::plan_memory(liveness);

    assert_no_conflicts(liveness, offsets);

    REQUIRE(total == 100);
}

TEST_CASE("plan_memory: fully concurrent lifetimes cannot share any memory") {
    auto a = dummy_tensor();
    auto b = dummy_tensor();
    auto c = dummy_tensor();

    std::unordered_map<std::shared_ptr<zerograd::Tensor>, std::tuple<std::size_t,std::size_t,std::size_t>> liveness{
        {a, {0, 10, 50}},
        {b, {0, 10, 30}},
        {c, {0, 10, 20}},
    };

    auto [total, offsets] = zerograd::MemoryPlanner::plan_memory(liveness);

    assert_no_conflicts(liveness, offsets);

    REQUIRE(total == 100);
}

TEST_CASE("plan_memory: touching boundaries (death == birth) count as non-overlapping") {
    auto a = dummy_tensor();
    auto b = dummy_tensor();

    std::unordered_map<std::shared_ptr<zerograd::Tensor>, std::tuple<std::size_t,std::size_t,std::size_t>> liveness{
        {a, {0, 5, 64}},
        {b, {5, 10, 64}},
    };

    auto [total, offsets] = zerograd::MemoryPlanner::plan_memory(liveness);

    assert_no_conflicts(liveness, offsets);
    REQUIRE(total == 64);
}

TEST_CASE("plan_memory: regression - two disjoint already-placed ranges both conflicting with a third") {
    auto P = dummy_tensor();
    auto Q = dummy_tensor();
    auto tA = dummy_tensor();

    std::unordered_map<std::shared_ptr<zerograd::Tensor>, std::tuple<std::size_t,std::size_t,std::size_t>> liveness{
        {P,  {0, 10, 5}},
        {Q,  {0, 10, 5}},
        {tA, {0, 10, 10}},
    };

    auto [total, offsets] = zerograd::MemoryPlanner::plan_memory(liveness);

    assert_no_conflicts(liveness, offsets);
}

TEST_CASE("plan_memory: mixed overlapping and non-overlapping tensors of varying sizes") {
    auto big1 = dummy_tensor();
    auto big2 = dummy_tensor();
    auto small1 = dummy_tensor();
    auto small2 = dummy_tensor();

    std::unordered_map<std::shared_ptr<zerograd::Tensor>, std::tuple<std::size_t,std::size_t,std::size_t>> liveness{
        {big1,   {0, 20, 1000}},
        {big2,   {5, 25, 800}},
        {small1, {0, 3, 50}},
        {small2, {3, 6, 50}},
    };

    auto [total, offsets] = zerograd::MemoryPlanner::plan_memory(liveness);

    assert_no_conflicts(liveness, offsets);

    REQUIRE(total >= 1000);
    REQUIRE(total <= 1000 + 800 + 50 + 50);
}

TEST_CASE("plan_memory: equal-size tensors produce a deterministic layout") {
    auto a = dummy_tensor();
    auto b = dummy_tensor();

    std::unordered_map<std::shared_ptr<zerograd::Tensor>, std::tuple<std::size_t,std::size_t,std::size_t>> liveness{
        {a, {0, 10, 64}},
        {b, {2, 10, 64}},
    };

    auto [total1, offsets1] = zerograd::MemoryPlanner::plan_memory(liveness);
    auto [total2, offsets2] = zerograd::MemoryPlanner::plan_memory(liveness);

    REQUIRE(total1 == total2);
    REQUIRE(offsets1.at(a) == offsets2.at(a));
    REQUIRE(offsets1.at(b) == offsets2.at(b));
}

TEST_CASE("plan_memory: total arena size matches the actual computed peak, not an overestimate") {
    auto a = dummy_tensor();
    auto b = dummy_tensor();

    std::unordered_map<std::shared_ptr<zerograd::Tensor>, std::tuple<std::size_t,std::size_t,std::size_t>> liveness{
        {a, {0, 5, 200}},
        {b, {0, 5, 100}},
    };

    auto [total, offsets] = zerograd::MemoryPlanner::plan_memory(liveness);

    std::size_t computed_peak = 0;
    for (auto& [t, off] : offsets) {
        std::size_t size = std::get<2>(liveness.at(t));
        computed_peak = std::max(computed_peak, off + size);
    }

    REQUIRE(total == computed_peak);
}

TEST_CASE("plan_memory: optimal oracle benchmark on small graphs") {
    auto a = dummy_tensor();
    auto b = dummy_tensor();
    auto c = dummy_tensor();
    auto d = dummy_tensor();
    auto e = dummy_tensor();

    std::unordered_map<std::shared_ptr<zerograd::Tensor>, std::tuple<std::size_t,std::size_t,std::size_t>> liveness{
        {a, {0, 10, 100}},
        {b, {2, 8,  80}},
        {c, {5, 12, 60}},
        {d, {0, 4,  40}},
        {e, {8, 15, 90}},
    };

    auto [greedy_total, greedy_offsets] = zerograd::MemoryPlanner::plan_memory(liveness);
    auto [optimal_total, optimal_offsets] = solve_optimal(liveness);

    assert_no_conflicts(liveness, greedy_offsets);
    assert_no_conflicts(liveness, optimal_offsets);

    REQUIRE(greedy_total >= optimal_total);
}

TEST_CASE("plan_memory: demonstration where greedy is strictly suboptimal") {
    auto A = dummy_tensor();
    auto B = dummy_tensor();
    auto C = dummy_tensor();
    auto D = dummy_tensor();
    auto E = dummy_tensor();

    std::unordered_map<std::shared_ptr<zerograd::Tensor>, std::tuple<std::size_t,std::size_t,std::size_t>> liveness{
        {D, {0, 2, 81}},
        {E, {6, 8, 81}},
        {A, {0, 4, 80}},
        {B, {4, 6, 80}},
        {C, {2, 6, 70}},
    };

    auto [greedy_total, greedy_offsets] = zerograd::MemoryPlanner::plan_memory(liveness);
    auto [optimal_total, optimal_offsets] = solve_optimal(liveness);

    assert_no_conflicts(liveness, greedy_offsets);
    assert_no_conflicts(liveness, optimal_offsets);

    REQUIRE(greedy_total == 231);
    REQUIRE(optimal_total == 161);
    REQUIRE(greedy_total > optimal_total);
}