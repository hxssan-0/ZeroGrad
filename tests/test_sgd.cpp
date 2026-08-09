#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <zerograd/optimizer.hpp>
#include <zerograd/tensor.hpp>

TEST_CASE("SGD Optimizer", "[optimizer][sgd]") {

    SECTION("step() updates data correctly") {
        auto param1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f},
            std::vector<size_t>{3}, true
        );
        auto param2 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{10.0f, 20.0f},
            std::vector<size_t>{2}, true
        );

        param1->grad = {0.1f, 0.2f, 0.3f};
        param2->grad = {1.0f, 2.0f};

        float lr = 0.1f;
        zerograd::Optimizer optimizer({param1, param2}, lr);

        optimizer.step();

        REQUIRE(param1->data[0] == Catch::Approx(1.0f - 0.1f * 0.1f));
        REQUIRE(param1->data[1] == Catch::Approx(2.0f - 0.1f * 0.2f));
        REQUIRE(param1->data[2] == Catch::Approx(3.0f - 0.1f * 0.3f));

        REQUIRE(param2->data[0] == Catch::Approx(10.0f - 0.1f * 1.0f));
        REQUIRE(param2->data[1] == Catch::Approx(20.0f - 0.1f * 2.0f));
    }

    SECTION("zero_grad() resets all gradients to zero") {
        auto param1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f},
            std::vector<size_t>{2}, true
        );
        auto param2 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{3.0f, 4.0f, 5.0f},
            std::vector<size_t>{3}, true
        );

        param1->grad = {5.0f, 6.0f};
        param2->grad = {7.0f, 8.0f, 9.0f};

        zerograd::Optimizer optimizer({param1, param2}, 0.01f);
        optimizer.zero_grad();

        for (float g : param1->grad)
            REQUIRE(g == Catch::Approx(0.0f));
        for (float g : param2->grad)
            REQUIRE(g == Catch::Approx(0.0f));
    }

    SECTION("Multiple steps accumulate correctly with zero_grad between") {
        auto param = std::make_shared<zerograd::Tensor>(
            std::vector<float>{10.0f},
            std::vector<size_t>{1}, true
        );

        zerograd::Optimizer optimizer({param}, 1.0f);

        param->grad = {2.0f};
        optimizer.step();
        REQUIRE(param->data[0] == Catch::Approx(8.0f));

        optimizer.zero_grad();
        REQUIRE(param->grad[0] == Catch::Approx(0.0f));

        param->grad = {3.0f};
        optimizer.step();
        REQUIRE(param->data[0] == Catch::Approx(5.0f));
    }

    SECTION("Zero learning rate - data unchanged") {
        auto param = std::make_shared<zerograd::Tensor>(
            std::vector<float>{5.0f, 10.0f},
            std::vector<size_t>{2}, true
        );
        param->grad = {100.0f, 200.0f};

        zerograd::Optimizer optimizer({param}, 0.0f);
        optimizer.step();

        REQUIRE(param->data[0] == Catch::Approx(5.0f));
        REQUIRE(param->data[1] == Catch::Approx(10.0f));
    }

    SECTION("Empty parameter list - no crash") {
        zerograd::Optimizer optimizer({}, 0.01f);
        REQUIRE_NOTHROW(optimizer.step());
        REQUIRE_NOTHROW(optimizer.zero_grad());
    }
}