#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <zerograd/tensor.h>
#include <cmath>
#include <functional>

inline float compute_tensor_numerical_gradient(
    const std::function<std::shared_ptr<zerograd::Tensor>()>& compute_graph,
    const std::shared_ptr<zerograd::Tensor>& target_tensor,
    std::size_t idx,
    float h = 1e-3f
)
{
    float original = target_tensor->data[idx];

    target_tensor->data[idx] = original + h;
    auto out_plus = compute_graph();
    float f_plus = out_plus->data[0];

    target_tensor->data[idx] = original - h;
    auto out_minus = compute_graph();
    float f_minus = out_minus->data[0];

    target_tensor->data[idx] = original;

    return (f_plus - f_minus) / (2.0f * h);
}

TEST_CASE("MSE Loss Forward", "[loss][mse]") {

    SECTION("Perfect prediction - loss is zero") {
        auto pred = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f},
            std::vector<size_t>{3}
        );
        auto target = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f},
            std::vector<size_t>{3}
        );

        auto loss = mse_loss(pred, target);
        REQUIRE(loss->data[0] == Catch::Approx(0.0f));
    }

    SECTION("Known values") {
        auto pred = std::make_shared<zerograd::Tensor>(
            std::vector<float>{2.0f, 4.0f},
            std::vector<size_t>{2}
        );
        auto target = std::make_shared<zerograd::Tensor>(
            std::vector<float>{0.0f, 0.0f},
            std::vector<size_t>{2}
        );

        auto loss = mse_loss(pred, target);
        REQUIRE(loss->data[0] == Catch::Approx(10.0f));
    }

    SECTION("Negative differences squared correctly") {
        auto pred = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 1.0f},
            std::vector<size_t>{2}
        );
        auto target = std::make_shared<zerograd::Tensor>(
            std::vector<float>{4.0f, 4.0f},
            std::vector<size_t>{2}
        );

        auto loss = mse_loss(pred, target);
        REQUIRE(loss->data[0] == Catch::Approx(9.0f));
    }

    SECTION("Loss is a scalar (0D) output") {
        auto pred = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
            std::vector<size_t>{2, 2}
        );
        auto target = std::make_shared<zerograd::Tensor>(
            std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f},
            std::vector<size_t>{2, 2}
        );

        auto loss = mse_loss(pred, target);
        REQUIRE(loss->shape == std::vector<size_t>{});
        REQUIRE(loss->data.size() == 1);
    }
}

TEST_CASE("MSE Loss Backward", "[loss][mse]") {

    SECTION("Gradient direction - overprediction pushes grad positive") {
        auto pred = std::make_shared<zerograd::Tensor>(
            std::vector<float>{5.0f},
            std::vector<size_t>{1}, true
        );
        auto target = std::make_shared<zerograd::Tensor>(
            std::vector<float>{2.0f},
            std::vector<size_t>{1}, true
        );

        auto loss = mse_loss(pred, target);
        loss->backward();

        REQUIRE(pred->grad[0] == Catch::Approx(6.0f));
    }

    SECTION("Gradient check - mse_loss") {
        auto pred = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.5f, -2.0f, 3.3f},
            std::vector<size_t>{3}, true
        );
        auto target = std::make_shared<zerograd::Tensor>(
            std::vector<float>{0.5f, 1.0f, -1.0f},
            std::vector<size_t>{3}, true
        );

        auto forward = [&pred, &target]() { return mse_loss(pred, target); };

        auto out = forward();
        out->backward();

        for (size_t i = 0; i < pred->data.size(); ++i) {
            float numerical = compute_tensor_numerical_gradient(forward, pred, i);
            REQUIRE(pred->grad[i] == Catch::Approx(numerical).margin(1e-3f));
        }
    }

    SECTION("Gradient check - target side too") {
        auto pred = std::make_shared<zerograd::Tensor>(
            std::vector<float>{2.0f, 3.0f},
            std::vector<size_t>{2}, true
        );
        auto target = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 5.0f},
            std::vector<size_t>{2}, true
        );

        auto forward = [&pred, &target]() { return mse_loss(pred, target); };

        auto out = forward();
        out->backward();

        for (size_t i = 0; i < target->data.size(); ++i) {
            float numerical = compute_tensor_numerical_gradient(forward, target, i);
            REQUIRE(target->grad[i] == Catch::Approx(numerical).margin(1e-3f));
        }
    }
}

TEST_CASE("Cross Entropy Loss Forward", "[loss][ce]") {

    SECTION("Confident correct prediction gives low loss") {
        auto logits = std::make_shared<zerograd::Tensor>(
            std::vector<float>{10.0f, 0.0f, 0.0f},
            std::vector<size_t>{1, 3}
        );
        auto loss = ce_loss(logits, {0});
        REQUIRE(loss->data[0] < 0.01f);
    }

    SECTION("Confident wrong prediction gives high loss") {
        auto logits = std::make_shared<zerograd::Tensor>(
            std::vector<float>{10.0f, 0.0f, 0.0f},
            std::vector<size_t>{1, 3}
        );
        auto loss = ce_loss(logits, {1});
        REQUIRE(loss->data[0] > 5.0f);
    }

    SECTION("Uniform logits give -log(1/num_classes)") {
        auto logits = std::make_shared<zerograd::Tensor>(
            std::vector<float>{0.0f, 0.0f, 0.0f},
            std::vector<size_t>{1, 3}
        );
        auto loss = ce_loss(logits, {0});
        REQUIRE(loss->data[0] == Catch::Approx(std::log(3.0f)).margin(1e-3f));
    }

    SECTION("Batch of 2 - loss is mean across batch") {
        auto logits = std::make_shared<zerograd::Tensor>(
            std::vector<float>{10.0f, 0.0f, 0.0f,
                                0.0f, 0.0f, 10.0f},
            std::vector<size_t>{2, 3}
        );
        
        auto loss = ce_loss(logits, {0, 2});
        REQUIRE(loss->data[0] < 0.01f);
    }

    SECTION("Loss is a scalar output") {
        auto logits = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f},
            std::vector<size_t>{1, 3}
        );
        auto loss = ce_loss(logits, {1});
        REQUIRE(loss->shape == std::vector<size_t>{});
    }

    SECTION("Wrong shape throws") {
        auto logits = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f},
            std::vector<size_t>{3}
        );
        REQUIRE_THROWS_AS(ce_loss(logits, {0}), std::runtime_error);
    }

    SECTION("Mismatched batch/targets size throws") {
        auto logits = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
            std::vector<size_t>{2, 3}
        );
        REQUIRE_THROWS_AS(ce_loss(logits, {0}), std::runtime_error);
    }

    SECTION("Numerical stability - large logits don't overflow") {
        auto logits = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1000.0f, 999.0f, 998.0f},
            std::vector<size_t>{1, 3}
        );
        auto loss = ce_loss(logits, {0});
        REQUIRE(std::isfinite(loss->data[0]));
    }
}

TEST_CASE("Cross Entropy Loss Backward - Rigorous Gradient Check", "[loss][ce][gradcheck]") {

    SECTION("Single example, multiple classes") {
        auto logits = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.2f, -0.5f, 2.3f, 0.1f},
            std::vector<size_t>{1, 4}, true
        );
        std::vector<std::size_t> targets = {2};

        auto forward = [&logits, &targets]() { return ce_loss(logits, targets); };

        auto out = forward();
        out->backward();

        for (size_t i = 0; i < logits->data.size(); ++i) {
            float numerical = compute_tensor_numerical_gradient(forward, logits, i);
            REQUIRE(logits->grad[i] == Catch::Approx(numerical).margin(1e-3f));
        }
    }

    SECTION("Batch of 3, different target classes") {
        auto logits = std::make_shared<zerograd::Tensor>(
            std::vector<float>{
                0.5f, 1.2f, -0.3f, 2.1f,
                -1.0f, 0.8f, 1.5f, 0.2f,
                2.0f, -0.5f, 0.3f, 1.1f
            },
            std::vector<size_t>{3, 4}, true
        );
        std::vector<std::size_t> targets = {3, 2, 0};

        auto forward = [&logits, &targets]() { return ce_loss(logits, targets); };

        auto out = forward();
        out->backward();

        for (size_t i = 0; i < logits->data.size(); ++i) {
            float numerical = compute_tensor_numerical_gradient(forward, logits, i);
            REQUIRE(logits->grad[i] == Catch::Approx(numerical).margin(1e-3f));
        }
    }

    SECTION("Extreme logit values - gradient check still holds") {
        auto logits = std::make_shared<zerograd::Tensor>(
            std::vector<float>{50.0f, -50.0f, 0.0f},
            std::vector<size_t>{1, 3}, true
        );
        std::vector<std::size_t> targets = {0};

        auto forward = [&logits, &targets]() { return ce_loss(logits, targets); };

        auto out = forward();
        out->backward();

        for (size_t i = 0; i < logits->data.size(); ++i) {
            float numerical = compute_tensor_numerical_gradient(forward, logits, i);
            REQUIRE(logits->grad[i] == Catch::Approx(numerical).margin(1e-2f));
        }
    }

    SECTION("Gradient matches closed-form softmax - target minus one-hot") {
        auto logits = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f},
            std::vector<size_t>{1, 3}, true
        );
        std::vector<std::size_t> targets = {1};

        auto loss = ce_loss(logits, targets);
        loss->backward();

        float max_val = 3.0f;
        float sum_exp = std::exp(1.0f - max_val) + std::exp(2.0f - max_val) + std::exp(3.0f - max_val);
        float s0 = std::exp(1.0f - max_val) / sum_exp;
        float s1 = std::exp(2.0f - max_val) / sum_exp;
        float s2 = std::exp(3.0f - max_val) / sum_exp;

        REQUIRE(logits->grad[0] == Catch::Approx(s0).margin(1e-4f));
        REQUIRE(logits->grad[1] == Catch::Approx(s1 - 1.0f).margin(1e-4f));
        REQUIRE(logits->grad[2] == Catch::Approx(s2).margin(1e-4f));
    }
}