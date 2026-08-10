#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <zerograd/tensor.hpp>
#include <zerograd/graph_serializer.hpp>
#include <vector>
#include <memory>
#include <numeric>
#include <cmath>
#include <fstream>

TEST_CASE("GraphSerializer::validate_dag accepts a normal acyclic graph") {
    auto t1 = std::make_shared<zerograd::Tensor>(
        std::vector<float>{1.0f, 2.0f}, std::vector<std::size_t>{2}, true);
    auto t2 = zerograd::exp(t1);
    auto t3 = zerograd::max(t2);

    REQUIRE(zerograd::GraphSerializer::validate_dag(t3) == true);
}

TEST_CASE("GraphSerializer::validate_dag rejects a forced cycle") {
    auto t1 = std::make_shared<zerograd::Tensor>(
        std::vector<float>{1.0f}, std::vector<std::size_t>{1}, true);
    auto t2 = zerograd::exp(t1);

    t1->_children.push_back(t2);

    REQUIRE(zerograd::GraphSerializer::validate_dag(t2) == false);
}

TEST_CASE("GraphSerializer::serialize writes expected node/edge structure") {
    auto t1 = std::make_shared<zerograd::Tensor>(
        std::vector<float>{1.0f, 2.0f}, std::vector<std::size_t>{2}, true);
    auto t2 = zerograd::exp(t1);
    auto t3 = zerograd::max(t2);

    std::string path = "test_graph_dump.json";
    zerograd::GraphSerializer::serialize(t3, path);

    std::ifstream file(path);
    REQUIRE(file.good());
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    REQUIRE(content.find("\"nodes\"") != std::string::npos);
    REQUIRE(content.find("\"edges\"") != std::string::npos);
    REQUIRE(content.find("\"op\": \"exp\"") != std::string::npos);
    REQUIRE(content.find("\"op\": \"max\"") != std::string::npos);

    std::remove(path.c_str());
}

TEST_CASE("GraphSerializer::serialize refuses to dump a cyclic graph") {
    auto t1 = std::make_shared<zerograd::Tensor>(
        std::vector<float>{1.0f}, std::vector<std::size_t>{1}, true);
    auto t2 = zerograd::exp(t1);
    t1->_children.push_back(t2);

    REQUIRE_THROWS_AS(zerograd::GraphSerializer::serialize(t2, "should_not_exist.json"), std::runtime_error);
}