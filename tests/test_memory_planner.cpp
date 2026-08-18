#include <zerograd/memory_planner.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <zerograd/tensor.hpp>

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