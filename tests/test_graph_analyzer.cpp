#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <zerograd/tensor.hpp>
#include <zerograd/graph_analyzer.hpp>
#include <vector>
#include <memory>
#include <numeric>
#include <cmath>

TEST_CASE("GraphAnalyzer: chain graph produces correct birth/death ordering") {
    auto x = std::make_shared<zerograd::Tensor>(
        std::vector<float>{1.0f, 2.0f}, std::vector<std::size_t>{2}, true);
    auto y = zerograd::exp(x);
    auto z = zerograd::max(y);

    zerograd::GraphAnalyzer analyzer;
    auto result = analyzer.dry_forward(z);

    REQUIRE(result.size() == 3);

    auto [x_birth, x_death, x_size] = result[x];
    auto [y_birth, y_death, y_size] = result[y];
    auto [z_birth, z_death, z_size] = result[z];

    REQUIRE(x_birth < y_birth);
    REQUIRE(y_birth < z_birth);

    REQUIRE(x_death > x_birth);
    REQUIRE(y_death > y_birth);
    REQUIRE(z_death > z_birth);

    REQUIRE(y_death == z_death);

    REQUIRE(x_death > y_death);
}

TEST_CASE("GraphAnalyzer: tensor used twice by the same op dies only after both uses") {
    auto x = std::make_shared<zerograd::Tensor>(
        std::vector<float>{3.0f}, std::vector<std::size_t>{1}, true);
    auto y = x + x;
    auto z = zerograd::max(y);

    zerograd::GraphAnalyzer analyzer;
    auto result = analyzer.dry_forward(z);

    REQUIRE(result.size() == 3);

    auto [x_birth, x_death, x_size] = result[x];
    auto [y_birth, y_death, y_size] = result[y];

    REQUIRE(x_death > y_death);
}

TEST_CASE("GraphAnalyzer: single node with no children still gets a valid interval") {
    auto x = std::make_shared<zerograd::Tensor>(
        std::vector<float>{5.0f}, std::vector<std::size_t>{1}, true);

    zerograd::GraphAnalyzer analyzer;
    auto result = analyzer.dry_forward(x);

    REQUIRE(result.size() == 1);
    auto [birth, death, size] = result[x];
    REQUIRE(death > birth);
}

TEST_CASE("GraphAnalyzer: size_bytes matches tensor data size") {
    auto x = std::make_shared<zerograd::Tensor>(
        std::vector<float>(20, 1.0f), std::vector<std::size_t>{4, 5}, true);
    auto y = zerograd::exp(x);

    zerograd::GraphAnalyzer analyzer;
    auto result = analyzer.dry_forward(y);

    auto [x_birth, x_death, x_size] = result[x];
    auto [y_birth, y_death, y_size] = result[y];

    REQUIRE(x_size == 20 * sizeof(float));
    REQUIRE(y_size == 20 * sizeof(float));
}

TEST_CASE("GraphAnalyzer: dry_forward does not mutate real ref_count") {
    auto x = std::make_shared<zerograd::Tensor>(
        std::vector<float>{1.0f}, std::vector<std::size_t>{1}, true);
    auto y = zerograd::exp(x);
    auto z = zerograd::max(y);

    std::size_t x_ref_before = x->ref_count;
    std::size_t y_ref_before = y->ref_count;

    zerograd::GraphAnalyzer analyzer;
    analyzer.dry_forward(z);

    REQUIRE(x->ref_count == x_ref_before);
    REQUIRE(y->ref_count == y_ref_before);
}

TEST_CASE("GraphAnalyzer: no tensor keeps the max_possible_death placeholder") {
    auto x = std::make_shared<zerograd::Tensor>(
        std::vector<float>{1.0f, 2.0f}, std::vector<std::size_t>{2}, true);
    auto y = zerograd::exp(x);
    auto z = zerograd::max(y);

    zerograd::GraphAnalyzer analyzer;
    auto result = analyzer.dry_forward(z);

    std::size_t topo_size = result.size();
    for (auto& [tensor, triple] : result) {
        auto [birth, death, size] = triple;
        REQUIRE(death <= birth + topo_size + 1);
    }
}