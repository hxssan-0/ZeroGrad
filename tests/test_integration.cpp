#include <catch2/catch_test_macros.hpp>
#include <zerograd/tensor.h>
#include <zerograd/linear.h>
#include <zerograd/sequential.h>
#include <zerograd/optimizer.h>
#include <zerograd/activations.h>
#include <memory>

TEST_CASE("Training loop check - loss decreases", "[integration]") {

    SECTION("Tiny MLP overfits a single MSE example") {
        zerograd::Sequential model;
        model.add(std::make_unique<zerograd::Linear>(2, 4));
        model.add(std::make_unique<zerograd::ReLU>());
        model.add(std::make_unique<zerograd::Linear>(4, 1));

        auto input = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f},
            std::vector<size_t>{1, 2}
        );
        auto target = std::make_shared<zerograd::Tensor>(
            std::vector<float>{5.0f},
            std::vector<size_t>{1, 1}
        );

        zerograd::Optimizer optimizer(model.parameters(), 0.01f);

        float first_loss = 0.0f;
        float last_loss = 0.0f;

        for (int step = 0; step < 50; ++step) {
            optimizer.zero_grad();
            auto pred = model.forward(input);
            auto loss = mse_loss(pred, target);
            loss->backward();
            optimizer.step();

            if (step == 0) first_loss = loss->data[0];
            if (step == 49) last_loss = loss->data[0];
        }

        REQUIRE(last_loss < first_loss);
    }

    SECTION("Tiny classifier overfits a single CE example") {
        zerograd::Sequential model;
        model.add(std::make_unique<zerograd::Linear>(3, 8));
        model.add(std::make_unique<zerograd::ReLU>());
        model.add(std::make_unique<zerograd::Linear>(8, 3));

        auto input = std::make_shared<zerograd::Tensor>(
            std::vector<float>{0.5f, -0.3f, 1.2f},
            std::vector<size_t>{1, 3}
        );
        std::vector<std::size_t> target = {1};

        zerograd::Optimizer optimizer(model.parameters(), 0.05f);

        float first_loss = 0.0f;
        float last_loss = 0.0f;

        for (int step = 0; step < 50; ++step) {
            optimizer.zero_grad();
            auto logits = model.forward(input);
            auto loss = ce_loss(logits, target);
            loss->backward();
            optimizer.step();

            if (step == 0) first_loss = loss->data[0];
            if (step == 49) last_loss = loss->data[0];
        }

        REQUIRE(last_loss < first_loss);
        REQUIRE(last_loss < 0.5f);
    }
}