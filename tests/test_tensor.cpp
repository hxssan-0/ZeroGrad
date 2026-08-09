#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <zerograd/tensor.hpp>
#include <vector>
#include <memory>
#include <numeric>
#include <cmath>

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

inline void zero_grad(const std::shared_ptr<zerograd::Tensor>& t)
{
    std::fill(t->grad.begin(), t->grad.end(), 0.0f);
}

TEST_CASE("Tensor Addition", "[tensor][forward][add]") {
    
    SECTION("Same Shape Addition (2, 2) + (2, 2)") {
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, 
            std::vector<size_t>{2, 2}
        );
        auto t2 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{10.0f, 10.0f, 10.0f, 10.0f}, 
            std::vector<size_t>{2, 2}
        );

        auto result = t1 + t2;

        REQUIRE(result->shape == std::vector<size_t>{2, 2});
        
        std::vector<float> expected = {11.0f, 12.0f, 13.0f, 14.0f};
        REQUIRE(result->data.size() == expected.size());
        
        for (size_t i = 0; i < expected.size(); ++i) {
            REQUIRE(result->data[i] == Catch::Approx(expected[i]));
        }
    }

    SECTION("Simple Broadcasting (2, 3) + (3,)") {
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f, 
                               4.0f, 5.0f, 6.0f}, 
            std::vector<size_t>{2, 3}
        );
        
        auto t2 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{10.0f, 100.0f, 1000.0f}, 
            std::vector<size_t>{3}
        );

        auto result = t1 + t2; 

        REQUIRE(result->shape == std::vector<size_t>{2, 3});
        
        std::vector<float> expected = {
            11.0f, 102.0f, 1003.0f, 
            14.0f, 105.0f, 1006.0f
        };
        
        for (size_t i = 0; i < expected.size(); ++i) {
            REQUIRE(result->data[i] == Catch::Approx(expected[i]));
        }
    }

    SECTION("Complex Broadcasting (3, 1) + (3,)") {
        
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f}, 
            std::vector<size_t>{3, 1}
        );
        
        auto t2 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{10.0f, 20.0f, 30.0f}, 
            std::vector<size_t>{3}
        );

        
        auto result = t1 + t2; 

        REQUIRE(result->shape == std::vector<size_t>{3, 3});
        
        std::vector<float> expected = {
            11.0f, 21.0f, 31.0f,
            12.0f, 22.0f, 32.0f,  
            13.0f, 23.0f, 33.0f   
        };
        
        for (size_t i = 0; i < expected.size(); ++i) {
            REQUIRE(result->data[i] == Catch::Approx(expected[i]));
        }
    }

    SECTION("Incompatible shapes throw") {
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f}, 
            std::vector<size_t>{3}
        );
        auto t2 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f}, 
            std::vector<size_t>{2}
        );
        REQUIRE_THROWS_AS(t1 + t2, std::invalid_argument);
    }

    SECTION("1D + 1D same size") {
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f}, 
            std::vector<size_t>{3}
        );
        auto t2 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{4.0f, 5.0f, 6.0f}, 
            std::vector<size_t>{3}
        );
        auto result = t1 + t2;
        REQUIRE(result->data == std::vector<float>{5.0f, 7.0f, 9.0f});
    }

    SECTION("Scalar (0D) + Scalar (0D)") {
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{3.0f}, 
            std::vector<size_t>{}
        );
        auto t2 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{4.0f}, 
            std::vector<size_t>{}
        );

        auto result = t1 + t2;

        REQUIRE(result->shape == std::vector<size_t>{});
        REQUIRE(result->data[0] == Catch::Approx(7.0f));
    }

    SECTION("Scalar (0D) + 1D tensor (3,)") {
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{2.0f}, 
            std::vector<size_t>{}
        );
        auto t2 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f}, 
            std::vector<size_t>{3}
        );

        auto result = t1 + t2;

        REQUIRE(result->shape == std::vector<size_t>{3});
        REQUIRE(result->data == std::vector<float>{3.0f, 4.0f, 5.0f});
    }
}

TEST_CASE("Tensor Subtraction", "[tensor][forward][sub]") {

    SECTION("Same Shape (2,2) - (2,2)") {
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{10.0f, 20.0f, 30.0f, 40.0f},
            std::vector<size_t>{2, 2}
        );
        auto t2 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
            std::vector<size_t>{2, 2}
        );

        auto result = t1 - t2;

        REQUIRE(result->shape == std::vector<size_t>{2, 2});
        std::vector<float> expected = {9.0f, 18.0f, 27.0f, 36.0f};
        for (size_t i = 0; i < expected.size(); ++i)
            REQUIRE(result->data[i] == Catch::Approx(expected[i]));
    }

    SECTION("Broadcasting (2,3) - (3,)") {
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{10.0f, 20.0f, 30.0f,
                               40.0f, 50.0f, 60.0f},
            std::vector<size_t>{2, 3}
        );
        auto t2 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f},
            std::vector<size_t>{3}
        );

        auto result = t1 - t2;

        REQUIRE(result->shape == std::vector<size_t>{2, 3});
        std::vector<float> expected = {
            9.0f, 18.0f, 27.0f,
            39.0f, 48.0f, 57.0f
        };
        for (size_t i = 0; i < expected.size(); ++i)
            REQUIRE(result->data[i] == Catch::Approx(expected[i]));
    }

    SECTION("Self Subtraction - result is zero") {
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f},
            std::vector<size_t>{3}
        );

        auto result = t1 - t1;

        for (size_t i = 0; i < 3; ++i)
            REQUIRE(result->data[i] == Catch::Approx(0.0f));
    }

    SECTION("Subtraction with negative values") {
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{-1.0f, -2.0f, -3.0f},
            std::vector<size_t>{3}
        );
        auto t2 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{-4.0f, -5.0f, -6.0f},
            std::vector<size_t>{3}
        );

        auto result = t1 - t2;

        std::vector<float> expected = {3.0f, 3.0f, 3.0f};
        for (size_t i = 0; i < expected.size(); ++i)
            REQUIRE(result->data[i] == Catch::Approx(expected[i]));
    }

    SECTION("Scalar (0D) - Scalar (0D)") {
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{5.0f},
            std::vector<size_t>{}
        );
        auto t2 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{3.0f},
            std::vector<size_t>{}
        );

        auto result = t1 - t2;

        REQUIRE(result->shape == std::vector<size_t>{});
        REQUIRE(result->data[0] == Catch::Approx(2.0f));
    }

    SECTION("Incompatible shapes throw") {
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f},
            std::vector<size_t>{3}
        );
        auto t2 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f},
            std::vector<size_t>{2}
        );
        REQUIRE_THROWS_AS(t1 - t2, std::invalid_argument);
    }
}

TEST_CASE("Tensor Multiplication", "[tensor][forward][mul]") {

    SECTION("Same Shape (2,2) * (2,2)") {
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
            std::vector<size_t>{2, 2}
        );
        auto t2 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{2.0f, 3.0f, 4.0f, 5.0f},
            std::vector<size_t>{2, 2}
        );

        auto result = t1 * t2;

        REQUIRE(result->shape == std::vector<size_t>{2, 2});
        std::vector<float> expected = {2.0f, 6.0f, 12.0f, 20.0f};
        for (size_t i = 0; i < expected.size(); ++i)
            REQUIRE(result->data[i] == Catch::Approx(expected[i]));
    }

    SECTION("Broadcasting (2,3) * (3,)") {
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f},
            std::vector<size_t>{2, 3}
        );
        auto t2 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{2.0f, 3.0f, 4.0f},
            std::vector<size_t>{3}
        );

        auto result = t1 * t2;

        REQUIRE(result->shape == std::vector<size_t>{2, 3});
        std::vector<float> expected = {
            2.0f,  6.0f,  12.0f,
            8.0f, 15.0f,  24.0f
        };
        for (size_t i = 0; i < expected.size(); ++i)
            REQUIRE(result->data[i] == Catch::Approx(expected[i]));
    }

    SECTION("Multiply by zero") {
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
            std::vector<size_t>{4}
        );
        auto t2 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f},
            std::vector<size_t>{4}
        );

        auto result = t1 * t2;

        for (size_t i = 0; i < 4; ++i)
            REQUIRE(result->data[i] == Catch::Approx(0.0f));
    }

    SECTION("Multiply by one") {
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
            std::vector<size_t>{4}
        );
        auto t2 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f},
            std::vector<size_t>{4}
        );

        auto result = t1 * t2;

        for (size_t i = 0; i < 4; ++i)
            REQUIRE(result->data[i] == Catch::Approx(t1->data[i]));
    }

    SECTION("Multiply by negative") {
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, -2.0f, 3.0f, -4.0f},
            std::vector<size_t>{4}
        );
        auto t2 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{-1.0f, -1.0f, -1.0f, -1.0f},
            std::vector<size_t>{4}
        );

        auto result = t1 * t2;

        std::vector<float> expected = {-1.0f, 2.0f, -3.0f, 4.0f};
        for (size_t i = 0; i < expected.size(); ++i)
            REQUIRE(result->data[i] == Catch::Approx(expected[i]));
    }

    SECTION("Scalar (0D) * Scalar (0D)") {
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{3.0f},
            std::vector<size_t>{}
        );
        auto t2 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{4.0f},
            std::vector<size_t>{}
        );

        auto result = t1 * t2;

        REQUIRE(result->shape == std::vector<size_t>{});
        REQUIRE(result->data[0] == Catch::Approx(12.0f));
    }

    SECTION("Scalar (0D) * 1D tensor") {
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{2.0f},
            std::vector<size_t>{}
        );
        auto t2 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f},
            std::vector<size_t>{3}
        );

        auto result = t1 * t2;

        REQUIRE(result->shape == std::vector<size_t>{3});
        std::vector<float> expected = {2.0f, 4.0f, 6.0f};
        for (size_t i = 0; i < expected.size(); ++i)
            REQUIRE(result->data[i] == Catch::Approx(expected[i]));
    }

    SECTION("Incompatible shapes throw") {
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f},
            std::vector<size_t>{3}
        );
        auto t2 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f},
            std::vector<size_t>{2}
        );
        REQUIRE_THROWS_AS(t1 * t2, std::invalid_argument);
    }
}

TEST_CASE("Tensor Matmul", "[tensor][forward][matmul]") {

    SECTION("Square matrices (2,2) @ (2,2)") {
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
            std::vector<size_t>{2, 2}
        );
        auto t2 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{5.0f, 6.0f, 7.0f, 8.0f},
            std::vector<size_t>{2, 2}
        );

        auto result = matmul(t1, t2);

        REQUIRE(result->shape == std::vector<size_t>{2, 2});
        std::vector<float> expected = {19.0f, 22.0f, 43.0f, 50.0f};
        for (size_t i = 0; i < expected.size(); ++i)
            REQUIRE(result->data[i] == Catch::Approx(expected[i]));
    }

    SECTION("Rectangular (2,3) @ (3,2)") {
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f},
            std::vector<size_t>{2, 3}
        );
        auto t2 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{7.0f,  8.0f,
                               9.0f,  10.0f,
                               11.0f, 12.0f},
            std::vector<size_t>{3, 2}
        );

        auto result = matmul(t1, t2);

        REQUIRE(result->shape == std::vector<size_t>{2, 2});
        // row 0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // row 1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        std::vector<float> expected = {58.0f, 64.0f, 139.0f, 154.0f};
        for (size_t i = 0; i < expected.size(); ++i)
            REQUIRE(result->data[i] == Catch::Approx(expected[i]));
    }

    SECTION("Rectangular (3,2) @ (2,4)") {
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f,
                               3.0f, 4.0f,
                               5.0f, 6.0f},
            std::vector<size_t>{3, 2}
        );
        auto t2 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f,
                               5.0f, 6.0f, 7.0f, 8.0f},
            std::vector<size_t>{2, 4}
        );

        auto result = matmul(t1, t2);

        REQUIRE(result->shape == std::vector<size_t>{3, 4});
        std::vector<float> expected = {
            11.0f, 14.0f, 17.0f, 20.0f,
            23.0f, 30.0f, 37.0f, 44.0f,
            35.0f, 46.0f, 57.0f, 68.0f
        };
        for (size_t i = 0; i < expected.size(); ++i)
            REQUIRE(result->data[i] == Catch::Approx(expected[i]));
    }

    SECTION("Identity matrix - result equals input") {
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f},
            std::vector<size_t>{2, 3}
        );
        auto identity = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 0.0f, 0.0f,
                               0.0f, 1.0f, 0.0f,
                               0.0f, 0.0f, 1.0f},
            std::vector<size_t>{3, 3}
        );

        auto result = matmul(t1, identity);

        REQUIRE(result->shape == std::vector<size_t>{2, 3});
        for (size_t i = 0; i < t1->data.size(); ++i)
            REQUIRE(result->data[i] == Catch::Approx(t1->data[i]));
    }

    SECTION("Multiply by zero matrix - result is zero") {
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
            std::vector<size_t>{2, 2}
        );
        auto zeros = std::make_shared<zerograd::Tensor>(
            std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f},
            std::vector<size_t>{2, 2}
        );

        auto result = matmul(t1, zeros);

        for (size_t i = 0; i < result->data.size(); ++i)
            REQUIRE(result->data[i] == Catch::Approx(0.0f));
    }

    SECTION("Batched (2,3,4) @ (2,4,5)") {
        // batch of 2, each (3,4) @ (4,5) = (3,5)
        std::vector<float> left_data(2 * 3 * 4);
        std::vector<float> right_data(2 * 4 * 5);
        std::iota(left_data.begin(), left_data.end(), 1.0f);
        std::iota(right_data.begin(), right_data.end(), 1.0f);

        auto t1 = std::make_shared<zerograd::Tensor>(left_data, std::vector<size_t>{2, 3, 4});
        auto t2 = std::make_shared<zerograd::Tensor>(right_data, std::vector<size_t>{2, 4, 5});

        auto result = matmul(t1, t2);

        REQUIRE(result->shape == std::vector<size_t>{2, 3, 5});
        REQUIRE(result->data.size() == 2 * 3 * 5);
    }

    SECTION("Batched with broadcasting (1,3,4) @ (2,4,5)") {
        std::vector<float> left_data(1 * 3 * 4, 1.0f);
        std::vector<float> right_data(2 * 4 * 5, 1.0f);

        auto t1 = std::make_shared<zerograd::Tensor>(left_data, std::vector<size_t>{1, 3, 4});
        auto t2 = std::make_shared<zerograd::Tensor>(right_data, std::vector<size_t>{2, 4, 5});

        auto result = matmul(t1, t2);

        // batch dim broadcasts: (1,3,4) @ (2,4,5) = (2,3,5)
        REQUIRE(result->shape == std::vector<size_t>{2, 3, 5});
        // each element should be 4 (sum of 4 ones)
        for (size_t i = 0; i < result->data.size(); ++i)
            REQUIRE(result->data[i] == Catch::Approx(4.0f));
    }

    SECTION("Vector dot product (1,N) @ (N,1) = (1,1)") {
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f},
            std::vector<size_t>{1, 3}
        );
        auto t2 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{4.0f, 5.0f, 6.0f},
            std::vector<size_t>{3, 1}
        );

        auto result = matmul(t1, t2);

        REQUIRE(result->shape == std::vector<size_t>{1, 1});
        // 1*4 + 2*5 + 3*6 = 32
        REQUIRE(result->data[0] == Catch::Approx(32.0f));
    }

    SECTION("Outer product (N,1) @ (1,M) = (N,M)") {
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f},
            std::vector<size_t>{3, 1}
        );
        auto t2 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{4.0f, 5.0f, 6.0f},
            std::vector<size_t>{1, 3}
        );

        auto result = matmul(t1, t2);

        REQUIRE(result->shape == std::vector<size_t>{3, 3});
        std::vector<float> expected = {
            4.0f,  5.0f,  6.0f,
            8.0f,  10.0f, 12.0f,
            12.0f, 15.0f, 18.0f
        };
        for (size_t i = 0; i < expected.size(); ++i)
            REQUIRE(result->data[i] == Catch::Approx(expected[i]));
    }

    SECTION("Incompatible inner dims throw") {
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
            std::vector<size_t>{2, 3}
        );
        auto t2 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
            std::vector<size_t>{2, 2}
        );
        REQUIRE_THROWS_AS(matmul(t1, t2), std::runtime_error);
    }

    SECTION("Rank < 2 throws") {
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f},
            std::vector<size_t>{3}
        );
        auto t2 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f},
            std::vector<size_t>{3}
        );
        REQUIRE_THROWS_AS(matmul(t1, t2), std::runtime_error);
    }
}

TEST_CASE("Tensor ReLU Forward", "[tensor][forward][relu]") {

    SECTION("Positive values pass through") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f},
            std::vector<size_t>{3}
        );
        auto result = relu(t);
        for (size_t i = 0; i < 3; ++i)
            REQUIRE(result->data[i] == Catch::Approx(t->data[i]));
    }

    SECTION("Negative values become zero") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{-1.0f, -2.0f, -3.0f},
            std::vector<size_t>{3}
        );
        auto result = relu(t);
        for (size_t i = 0; i < 3; ++i)
            REQUIRE(result->data[i] == Catch::Approx(0.0f));
    }

    SECTION("Mixed values") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{-1.0f, 0.0f, 1.0f, -2.0f, 3.0f},
            std::vector<size_t>{5}
        );
        auto result = relu(t);
        std::vector<float> expected = {0.0f, 0.0f, 1.0f, 0.0f, 3.0f};
        for (size_t i = 0; i < expected.size(); ++i)
            REQUIRE(result->data[i] == Catch::Approx(expected[i]));
    }

    SECTION("2D tensor") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{-1.0f, 2.0f, -3.0f, 4.0f},
            std::vector<size_t>{2, 2}
        );
        auto result = relu(t);
        std::vector<float> expected = {0.0f, 2.0f, 0.0f, 4.0f};
        for (size_t i = 0; i < expected.size(); ++i)
            REQUIRE(result->data[i] == Catch::Approx(expected[i]));
        REQUIRE(result->shape == std::vector<size_t>{2, 2});
    }
}

TEST_CASE("Tensor ReLU Backward", "[tensor][backward][relu]") {

    SECTION("Gradient passes through for positive, zero for negative") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{-1.0f, 2.0f, -3.0f, 4.0f},
            std::vector<size_t>{4}, true
        );
        auto result = relu(t);
        auto loss = sum(result);
        loss->backward();

        REQUIRE(t->grad[0] == Catch::Approx(0.0f));
        REQUIRE(t->grad[1] == Catch::Approx(1.0f));
        REQUIRE(t->grad[2] == Catch::Approx(0.0f));
        REQUIRE(t->grad[3] == Catch::Approx(1.0f));
    }

    SECTION("Gradient check - relu") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{0.5f, 1.5f, 2.0f},
            std::vector<size_t>{3}, true
        );

        auto forward = [&t]() { return sum(relu(t)); };

        auto out = forward();
        out->backward();

        for (size_t i = 0; i < t->data.size(); ++i) {
            float numerical = compute_tensor_numerical_gradient(forward, t, i);
            REQUIRE(t->grad[i] == Catch::Approx(numerical).margin(1e-3f));
        }
    }
}

// ─── SIGMOID ─────────────────────────────────────────────────────────────────

TEST_CASE("Tensor Sigmoid Forward", "[tensor][forward][sigmoid]") {

    SECTION("Output in range (0, 1)") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{-2.0f, -1.0f, 0.0f, 1.0f, 2.0f},
            std::vector<size_t>{5}
        );
        auto result = sigmoid(t);
        for (size_t i = 0; i < result->data.size(); ++i) {
            REQUIRE(result->data[i] > 0.0f);
            REQUIRE(result->data[i] < 1.0f);
        }
    }

    SECTION("sigmoid(0) = 0.5") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{0.0f},
            std::vector<size_t>{1}
        );
        auto result = sigmoid(t);
        REQUIRE(result->data[0] == Catch::Approx(0.5f));
    }

    SECTION("Known values") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, -1.0f},
            std::vector<size_t>{2}
        );
        auto result = sigmoid(t);
        REQUIRE(result->data[0] == Catch::Approx(1.0f / (1.0f + std::exp(-1.0f))));
        REQUIRE(result->data[1] == Catch::Approx(1.0f / (1.0f + std::exp(1.0f))));
    }

    SECTION("Shape preserved") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
            std::vector<size_t>{2, 2}
        );
        auto result = sigmoid(t);
        REQUIRE(result->shape == std::vector<size_t>{2, 2});
    }
}

TEST_CASE("Tensor Sigmoid Backward", "[tensor][backward][sigmoid]") {

    SECTION("Gradient check - sigmoid") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{-1.0f, 0.0f, 1.0f},
            std::vector<size_t>{3}, true
        );

        auto forward = [&t]() { return sum(sigmoid(t)); };

        auto out = forward();
        out->backward();

        for (size_t i = 0; i < t->data.size(); ++i) {
            float numerical = compute_tensor_numerical_gradient(forward, t, i);
            REQUIRE(t->grad[i] == Catch::Approx(numerical).margin(1e-3f));
        }
    }
}

// ─── TANH ────────────────────────────────────────────────────────────────────

TEST_CASE("Tensor Tanh Forward", "[tensor][forward][tanh]") {

    SECTION("tanh(0) = 0") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{0.0f},
            std::vector<size_t>{1}
        );
        auto result = tanh(t);
        REQUIRE(result->data[0] == Catch::Approx(0.0f));
    }

    SECTION("Output in range (-1, 1)") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{-5.0f, -1.0f, 0.0f, 1.0f, 5.0f},
            std::vector<size_t>{5}
        );
        auto result = tanh(t);
        for (size_t i = 0; i < result->data.size(); ++i) {
            REQUIRE(result->data[i] > -1.0f);
            REQUIRE(result->data[i] < 1.0f);
        }
    }

    SECTION("Known values") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, -1.0f, 0.5f},
            std::vector<size_t>{3}
        );
        auto result = tanh(t);
        for (size_t i = 0; i < t->data.size(); ++i)
            REQUIRE(result->data[i] == Catch::Approx(std::tanh(t->data[i])));
    }

    SECTION("Shape preserved") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
            std::vector<size_t>{2, 2}
        );
        auto result = tanh(t);
        REQUIRE(result->shape == std::vector<size_t>{2, 2});
    }
}

TEST_CASE("Tensor Tanh Backward", "[tensor][backward][tanh]") {

    SECTION("Gradient check - tanh") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{-0.5f, 0.0f, 0.5f},
            std::vector<size_t>{3}, true
        );

        auto forward = [&t]() { return sum(tanh(t)); };

        auto out = forward();
        out->backward();

        for (size_t i = 0; i < t->data.size(); ++i) {
            float numerical = compute_tensor_numerical_gradient(forward, t, i);
            REQUIRE(t->grad[i] == Catch::Approx(numerical).margin(1e-3f));
        }
    }
}

// ─── SUM ─────────────────────────────────────────────────────────────────────

TEST_CASE("Tensor Sum Forward", "[tensor][forward][sum]") {

    SECTION("1D sum") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
            std::vector<size_t>{4}
        );
        auto result = sum(t);
        REQUIRE(result->data[0] == Catch::Approx(10.0f));
        REQUIRE(result->shape == std::vector<size_t>{});
    }

    SECTION("2D sum") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
            std::vector<size_t>{2, 2}
        );
        auto result = sum(t);
        REQUIRE(result->data[0] == Catch::Approx(10.0f));
    }

    SECTION("Sum of negatives") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{-1.0f, -2.0f, -3.0f},
            std::vector<size_t>{3}
        );
        auto result = sum(t);
        REQUIRE(result->data[0] == Catch::Approx(-6.0f));
    }

    SECTION("Sum of zeros") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{0.0f, 0.0f, 0.0f},
            std::vector<size_t>{3}
        );
        auto result = sum(t);
        REQUIRE(result->data[0] == Catch::Approx(0.0f));
    }
}

TEST_CASE("Tensor Sum Backward", "[tensor][backward][sum]") {

    SECTION("Gradient flows equally to all elements") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
            std::vector<size_t>{4}, true
        );
        auto result = sum(t);
        result->backward();

        for (size_t i = 0; i < t->data.size(); ++i)
            REQUIRE(t->grad[i] == Catch::Approx(1.0f));
    }

    SECTION("Gradient check - sum") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f},
            std::vector<size_t>{3}, true
        );

        auto forward = [&t]() { return sum(t); };

        auto out = forward();
        out->backward();

        for (size_t i = 0; i < t->data.size(); ++i) {
            float numerical = compute_tensor_numerical_gradient(forward, t, i);
            REQUIRE(t->grad[i] == Catch::Approx(numerical).margin(1e-3f));
        }
    }
}

// ─── MEAN ────────────────────────────────────────────────────────────────────

TEST_CASE("Tensor Mean Forward", "[tensor][forward][mean]") {

    SECTION("1D mean") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
            std::vector<size_t>{4}
        );
        auto result = mean(t);
        REQUIRE(result->data[0] == Catch::Approx(2.5f));
        REQUIRE(result->shape == std::vector<size_t>{});
    }

    SECTION("2D mean") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
            std::vector<size_t>{2, 2}
        );
        auto result = mean(t);
        REQUIRE(result->data[0] == Catch::Approx(2.5f));
    }

    SECTION("Mean of identical values") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{3.0f, 3.0f, 3.0f},
            std::vector<size_t>{3}
        );
        auto result = mean(t);
        REQUIRE(result->data[0] == Catch::Approx(3.0f));
    }
}

TEST_CASE("Tensor Mean Backward", "[tensor][backward][mean]") {

    SECTION("Gradient divided equally") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
            std::vector<size_t>{4}, true
        );
        auto result = mean(t);
        result->backward();

        for (size_t i = 0; i < t->data.size(); ++i)
            REQUIRE(t->grad[i] == Catch::Approx(0.25f));
    }

    SECTION("Gradient check - mean") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f},
            std::vector<size_t>{3}, true
        );

        auto forward = [&t]() { return mean(t); };

        auto out = forward();
        out->backward();

        for (size_t i = 0; i < t->data.size(); ++i) {
            float numerical = compute_tensor_numerical_gradient(forward, t, i);
            REQUIRE(t->grad[i] == Catch::Approx(numerical).margin(1e-3f));
        }
    }
}

// ─── INTEGRATION ─────────────────────────────────────────────────────────────

TEST_CASE("Tensor Backward Integration", "[tensor][backward][integration]") {

    SECTION("Chain: matmul -> relu -> sum") {
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, -1.0f, 2.0f, -2.0f},
            std::vector<size_t>{2, 2}, true
        );
        auto t2 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f},
            std::vector<size_t>{2, 2}, true
        );

        auto forward = [&t1, &t2]() {
            return sum(relu(matmul(t1, t2)));
        };

        auto out = forward();
        out->backward();

        for (size_t i = 0; i < t1->data.size(); ++i) {
            float numerical = compute_tensor_numerical_gradient(forward, t1, i);
            REQUIRE(t1->grad[i] == Catch::Approx(numerical).margin(1e-3f));
        }
    }

    SECTION("Chain: matmul -> tanh -> mean") {
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{0.5f, -0.5f, 1.0f, -1.0f},
            std::vector<size_t>{2, 2}, true
        );
        auto t2 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
            std::vector<size_t>{2, 2}, true
        );

        auto forward = [&t1, &t2]() {
            return mean(tanh(matmul(t1, t2)));
        };

        auto out = forward();
        out->backward();

        for (size_t i = 0; i < t1->data.size(); ++i) {
            float numerical = compute_tensor_numerical_gradient(forward, t1, i);
            REQUIRE(t1->grad[i] == Catch::Approx(numerical).margin(1e-3f));
        }
    }

    SECTION("requires_grad=false - grads stay zero throughout chain") {
        auto t1 = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
            std::vector<size_t>{2, 2}, false
        );

        auto result = sum(relu(t1));
        result->backward();

        for (size_t i = 0; i < t1->grad.size(); ++i)
            REQUIRE(t1->grad[i] == Catch::Approx(0.0f));
    }
}

TEST_CASE("Broadcast Backward Gradient Check", "[tensor][backward][broadcast]") {
    auto t1 = std::make_shared<zerograd::Tensor>(
        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
        std::vector<size_t>{2, 3}, true
    );
    auto t2 = std::make_shared<zerograd::Tensor>(
        std::vector<float>{0.5f, -0.3f, 1.2f},
        std::vector<size_t>{3}, true
    );

    auto forward = [&t1, &t2]() { return sum(t1 * t2); };

    auto out = forward();
    out->backward();

    for (size_t i = 0; i < t2->data.size(); ++i) {
        float numerical = compute_tensor_numerical_gradient(forward, t2, i);
        REQUIRE(t2->grad[i] == Catch::Approx(numerical).margin(1e-3f));
    }
}


TEST_CASE("Tensor Max Forward", "[tensor][forward][max]") {

    SECTION("Simple max") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 5.0f, 3.0f},
            std::vector<size_t>{3}
        );
        auto result = max(t);
        REQUIRE(result->data[0] == Catch::Approx(5.0f));
        REQUIRE(result->shape == std::vector<size_t>{});
    }

    SECTION("Max with negative values") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{-5.0f, -1.0f, -3.0f},
            std::vector<size_t>{3}
        );
        auto result = max(t);
        REQUIRE(result->data[0] == Catch::Approx(-1.0f));
    }

    SECTION("Max at first index") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{10.0f, 2.0f, 3.0f},
            std::vector<size_t>{3}
        );
        auto result = max(t);
        REQUIRE(result->data[0] == Catch::Approx(10.0f));
    }

    SECTION("Tied max - takes first occurrence") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{5.0f, 5.0f, 3.0f},
            std::vector<size_t>{3}, true
        );
        auto result = max(t);
        result->backward();

        REQUIRE(t->grad[0] == Catch::Approx(1.0f));
        REQUIRE(t->grad[1] == Catch::Approx(0.0f));
    }
}

TEST_CASE("Tensor Max Backward", "[tensor][backward][max]") {

    SECTION("Gradient routes only to max element") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 5.0f, 3.0f, 2.0f},
            std::vector<size_t>{4}, true
        );
        auto result = max(t);
        result->backward();

        REQUIRE(t->grad[0] == Catch::Approx(0.0f));
        REQUIRE(t->grad[1] == Catch::Approx(1.0f));
        REQUIRE(t->grad[2] == Catch::Approx(0.0f));
        REQUIRE(t->grad[3] == Catch::Approx(0.0f));
    }

    SECTION("Gradient check - max") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{0.5f, 2.5f, 1.5f},
            std::vector<size_t>{3}, true
        );

        auto forward = [&t]() { return max(t); };

        auto out = forward();
        out->backward();

        for (size_t i = 0; i < t->data.size(); ++i) {
            float numerical = compute_tensor_numerical_gradient(forward, t, i);
            REQUIRE(t->grad[i] == Catch::Approx(numerical).margin(1e-3f));
        }
    }
}

TEST_CASE("Tensor Log Forward", "[tensor][forward][log]") {

    SECTION("log(1) = 0") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f},
            std::vector<size_t>{1}
        );
        auto result = log(t);
        REQUIRE(result->data[0] == Catch::Approx(0.0f).margin(1e-4f));
    }

    SECTION("Known values") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{std::exp(1.0f), std::exp(2.0f)},
            std::vector<size_t>{2}
        );
        auto result = log(t);
        REQUIRE(result->data[0] == Catch::Approx(1.0f).margin(1e-4f));
        REQUIRE(result->data[1] == Catch::Approx(2.0f).margin(1e-4f));
    }

    SECTION("Shape preserved") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
            std::vector<size_t>{2, 2}
        );
        auto result = log(t);
        REQUIRE(result->shape == std::vector<size_t>{2, 2});
    }

    SECTION("Near-zero input doesn't blow up (epsilon stability)") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{0.0f},
            std::vector<size_t>{1}
        );
        auto result = log(t);
        REQUIRE(std::isfinite(result->data[0]));
    }
}

TEST_CASE("Tensor Log Backward", "[tensor][backward][log]") {

    SECTION("Gradient check - log") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{0.5f, 1.5f, 3.0f},
            std::vector<size_t>{3}, true
        );

        auto forward = [&t]() { return sum(log(t)); };

        auto out = forward();
        out->backward();

        for (size_t i = 0; i < t->data.size(); ++i) {
            float numerical = compute_tensor_numerical_gradient(forward, t, i);
            REQUIRE(t->grad[i] == Catch::Approx(numerical).margin(1e-3f));
        }
    }
}

TEST_CASE("Tensor Exp Forward", "[tensor][forward][exp]") {

    SECTION("exp(0) = 1") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{0.0f},
            std::vector<size_t>{1}
        );
        auto result = exp(t);
        REQUIRE(result->data[0] == Catch::Approx(1.0f));
    }

    SECTION("Known values") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f},
            std::vector<size_t>{2}
        );
        auto result = exp(t);
        REQUIRE(result->data[0] == Catch::Approx(std::exp(1.0f)));
        REQUIRE(result->data[1] == Catch::Approx(std::exp(2.0f)));
    }

    SECTION("Negative input produces value between 0 and 1") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{-2.0f},
            std::vector<size_t>{1}
        );
        auto result = exp(t);
        REQUIRE(result->data[0] > 0.0f);
        REQUIRE(result->data[0] < 1.0f);
    }

    SECTION("Shape preserved") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
            std::vector<size_t>{2, 2}
        );
        auto result = exp(t);
        REQUIRE(result->shape == std::vector<size_t>{2, 2});
    }
}

TEST_CASE("Tensor Exp Backward", "[tensor][backward][exp]") {

    SECTION("Gradient check - exp") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{-1.0f, 0.5f, 1.5f},
            std::vector<size_t>{3}, true
        );

        auto forward = [&t]() { return sum(exp(t)); };

        auto out = forward();
        out->backward();

        for (size_t i = 0; i < t->data.size(); ++i) {
            float numerical = compute_tensor_numerical_gradient(forward, t, i);
            REQUIRE(t->grad[i] == Catch::Approx(numerical).margin(1e-3f));
        }
    }
}

TEST_CASE("Tensor Softmax Forward", "[tensor][forward][softmax]") {

    SECTION("Outputs sum to 1 - single row") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f},
            std::vector<size_t>{1, 3}
        );
        auto result = zerograd::softmax(t);

        float row_sum = 0.0f;
        for (float v : result->data) row_sum += v;
        REQUIRE(row_sum == Catch::Approx(1.0f).margin(1e-5f));
    }

    SECTION("Outputs sum to 1 - each row independently, batched") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f,
                               -1.0f, 0.0f, 5.0f,
                               10.0f, 10.0f, 10.0f},
            std::vector<size_t>{3, 3}
        );
        auto result = zerograd::softmax(t);

        for (size_t b = 0; b < 3; ++b) {
            float row_sum = 0.0f;
            for (size_t j = 0; j < 3; ++j)
                row_sum += result->data[b * 3 + j];
            REQUIRE(row_sum == Catch::Approx(1.0f).margin(1e-5f));
        }
    }

    SECTION("All outputs are positive") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{-5.0f, 0.0f, 5.0f, -100.0f},
            std::vector<size_t>{1, 4}
        );
        auto result = zerograd::softmax(t);
        for (float v : result->data)
            REQUIRE(v >= 0.0f);
    }

    SECTION("Equal inputs give uniform distribution") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{2.0f, 2.0f, 2.0f, 2.0f},
            std::vector<size_t>{1, 4}
        );
        auto result = zerograd::softmax(t);
        for (float v : result->data)
            REQUIRE(v == Catch::Approx(0.25f));
    }

    SECTION("Largest input gets largest probability") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 5.0f, 2.0f},
            std::vector<size_t>{1, 3}
        );
        auto result = zerograd::softmax(t);
        REQUIRE(result->data[1] > result->data[0]);
        REQUIRE(result->data[1] > result->data[2]);
    }

    SECTION("Numerical stability - large values don't overflow") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1000.0f, 999.0f, 998.0f},
            std::vector<size_t>{1, 3}
        );
        auto result = zerograd::softmax(t);
        for (float v : result->data)
            REQUIRE(std::isfinite(v));

        float row_sum = 0.0f;
        for (float v : result->data) row_sum += v;
        REQUIRE(row_sum == Catch::Approx(1.0f).margin(1e-4f));
    }

    SECTION("Numerical stability - very negative values don't underflow to nan") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{-1000.0f, -999.0f, -998.0f},
            std::vector<size_t>{1, 3}
        );
        auto result = zerograd::softmax(t);
        for (float v : result->data)
            REQUIRE(std::isfinite(v));
    }

    SECTION("Shape preserved") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
            std::vector<size_t>{2, 3}
        );
        auto result = zerograd::softmax(t);
        REQUIRE(result->shape == std::vector<size_t>{2, 3});
    }

    SECTION("Non-2D input throws") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f},
            std::vector<size_t>{3}
        );
        REQUIRE_THROWS_AS(zerograd::softmax(t), std::runtime_error);
    }
}

TEST_CASE("Tensor Softmax Backward", "[tensor][backward][softmax]") {

    SECTION("Gradient check - single row, multiple classes") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{0.5f, -1.2f, 2.3f, 0.1f},
            std::vector<size_t>{1, 4}, true
        );

        auto forward = [&t]() { return sum(zerograd::softmax(t)); };

        auto out = forward();
        out->backward();

        for (size_t i = 0; i < t->data.size(); ++i) {
            float numerical = compute_tensor_numerical_gradient(forward, t, i);
            REQUIRE(t->grad[i] == Catch::Approx(numerical).margin(1e-3f));
        }
    }

    SECTION("Gradient check - batched, multiple rows") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{
                1.0f, 2.0f, -1.0f,
                0.5f, -0.5f, 3.0f
            },
            std::vector<size_t>{2, 3}, true
        );

        auto forward = [&t]() { return sum(zerograd::softmax(t)); };

        auto out = forward();
        out->backward();

        for (size_t i = 0; i < t->data.size(); ++i) {
            float numerical = compute_tensor_numerical_gradient(forward, t, i);
            REQUIRE(t->grad[i] == Catch::Approx(numerical).margin(1e-3f));
        }
    }

    SECTION("Gradient check - composed with a non-uniform downstream weighting") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.5f, -0.5f, 2.0f},
            std::vector<size_t>{1, 3}, true
        );
        auto weights = std::make_shared<zerograd::Tensor>(
            std::vector<float>{2.0f, -1.0f, 0.5f},
            std::vector<size_t>{1, 3}
        );

        auto forward = [&t, &weights]() { return sum(zerograd::softmax(t) * weights); };

        auto out = forward();
        out->backward();

        for (size_t i = 0; i < t->data.size(); ++i) {
            float numerical = compute_tensor_numerical_gradient(forward, t, i);
            REQUIRE(t->grad[i] == Catch::Approx(numerical).margin(1e-3f));
        }
    }

    SECTION("Gradient check - extreme values (stability under backward too)") {
        auto t = std::make_shared<zerograd::Tensor>(
            std::vector<float>{50.0f, -50.0f, 0.0f},
            std::vector<size_t>{1, 3}, true
        );
        auto weights = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 1.0f, 1.0f},
            std::vector<size_t>{1, 3}
        );

        auto forward = [&t, &weights]() { return sum(zerograd::softmax(t) * weights); };

        auto out = forward();
        out->backward();

        for (size_t i = 0; i < t->data.size(); ++i) {
            REQUIRE(std::isfinite(t->grad[i]));
        }
    }
}

TEST_CASE("BatchNorm1d Forward", "[tensor][forward][batchnorm]") {

    SECTION("Training mode - output has ~zero mean, ~unit variance per feature") {
        auto input = std::make_shared<zerograd::Tensor>(
            std::vector<float>{
                1.0f, 10.0f,
                2.0f, 20.0f,
                3.0f, 30.0f,
                4.0f, 40.0f
            },
            std::vector<size_t>{4, 2}
        );
        auto gamma = std::make_shared<zerograd::Tensor>(std::vector<float>{1.0f, 1.0f}, std::vector<size_t>{2});
        auto beta = std::make_shared<zerograd::Tensor>(std::vector<float>{0.0f, 0.0f}, std::vector<size_t>{2});
        auto running_mean = std::make_shared<zerograd::Tensor>(std::vector<float>{0.0f, 0.0f}, std::vector<size_t>{2});
        auto running_var = std::make_shared<zerograd::Tensor>(std::vector<float>{1.0f, 1.0f}, std::vector<size_t>{2});

        auto result = zerograd::batchNorm1d(input, gamma, beta, running_mean, running_var, true);

        for (size_t feature = 0; feature < 2; ++feature) {
            float mean = 0.0f;
            for (size_t b = 0; b < 4; ++b) mean += result->data[b * 2 + feature];
            mean /= 4.0f;
            REQUIRE(mean == Catch::Approx(0.0f).margin(1e-4f));

            float var = 0.0f;
            for (size_t b = 0; b < 4; ++b) {
                float diff = result->data[b * 2 + feature] - mean;
                var += diff * diff;
            }
            var /= 4.0f;
            REQUIRE(var == Catch::Approx(1.0f).margin(1e-3f));
        }
    }

    SECTION("gamma=1, beta=0 preserves pure normalization") {
        auto input = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
            std::vector<size_t>{4, 1}
        );
        auto gamma = std::make_shared<zerograd::Tensor>(std::vector<float>{1.0f}, std::vector<size_t>{1});
        auto beta = std::make_shared<zerograd::Tensor>(std::vector<float>{0.0f}, std::vector<size_t>{1});
        auto running_mean = std::make_shared<zerograd::Tensor>(std::vector<float>{0.0f}, std::vector<size_t>{1});
        auto running_var = std::make_shared<zerograd::Tensor>(std::vector<float>{1.0f}, std::vector<size_t>{1});

        auto result = zerograd::batchNorm1d(input, gamma, beta, running_mean, running_var, true);

        REQUIRE(result->data[0] == Catch::Approx(-result->data[3]).margin(1e-4f));
        REQUIRE(result->data[1] == Catch::Approx(-result->data[2]).margin(1e-4f));
    }

    SECTION("gamma and beta shift/scale correctly") {
        auto input = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
            std::vector<size_t>{4, 1}
        );
        auto gamma = std::make_shared<zerograd::Tensor>(std::vector<float>{2.0f}, std::vector<size_t>{1});
        auto beta = std::make_shared<zerograd::Tensor>(std::vector<float>{5.0f}, std::vector<size_t>{1});
        auto running_mean = std::make_shared<zerograd::Tensor>(std::vector<float>{0.0f}, std::vector<size_t>{1});
        auto running_var = std::make_shared<zerograd::Tensor>(std::vector<float>{1.0f}, std::vector<size_t>{1});

        auto result = zerograd::batchNorm1d(input, gamma, beta, running_mean, running_var, true);

        float mean = 0.0f;
        for (float v : result->data) mean += v;
        mean /= result->data.size();
        REQUIRE(mean == Catch::Approx(5.0f).margin(1e-4f));
    }

    SECTION("Eval mode uses running stats, not batch stats") {
        auto input = std::make_shared<zerograd::Tensor>(
            std::vector<float>{100.0f, 200.0f},  // wildly different from running stats
            std::vector<size_t>{1, 2}
        );
        auto gamma = std::make_shared<zerograd::Tensor>(std::vector<float>{1.0f, 1.0f}, std::vector<size_t>{2});
        auto beta = std::make_shared<zerograd::Tensor>(std::vector<float>{0.0f, 0.0f}, std::vector<size_t>{2});
        auto running_mean = std::make_shared<zerograd::Tensor>(std::vector<float>{0.0f, 0.0f}, std::vector<size_t>{2});
        auto running_var = std::make_shared<zerograd::Tensor>(std::vector<float>{1.0f, 1.0f}, std::vector<size_t>{2});

        auto result = zerograd::batchNorm1d(input, gamma, beta, running_mean, running_var, false);

        REQUIRE(result->data[0] == Catch::Approx(100.0f).margin(0.1f));
        REQUIRE(result->data[1] == Catch::Approx(200.0f).margin(0.1f));
    }

    SECTION("Eval mode does not modify running stats") {
        auto input = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f},
            std::vector<size_t>{1, 2}
        );
        auto gamma = std::make_shared<zerograd::Tensor>(std::vector<float>{1.0f, 1.0f}, std::vector<size_t>{2});
        auto beta = std::make_shared<zerograd::Tensor>(std::vector<float>{0.0f, 0.0f}, std::vector<size_t>{2});
        auto running_mean = std::make_shared<zerograd::Tensor>(std::vector<float>{5.0f, 5.0f}, std::vector<size_t>{2});
        auto running_var = std::make_shared<zerograd::Tensor>(std::vector<float>{2.0f, 2.0f}, std::vector<size_t>{2});

        zerograd::batchNorm1d(input, gamma, beta, running_mean, running_var, false);

        REQUIRE(running_mean->data[0] == Catch::Approx(5.0f));
        REQUIRE(running_var->data[0] == Catch::Approx(2.0f));
    }

    SECTION("Training mode updates running stats using PyTorch momentum formula") {
        auto input = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
            std::vector<size_t>{4, 1}
        );
        auto gamma = std::make_shared<zerograd::Tensor>(std::vector<float>{1.0f}, std::vector<size_t>{1});
        auto beta = std::make_shared<zerograd::Tensor>(std::vector<float>{0.0f}, std::vector<size_t>{1});
        auto running_mean = std::make_shared<zerograd::Tensor>(std::vector<float>{0.0f}, std::vector<size_t>{1});
        auto running_var = std::make_shared<zerograd::Tensor>(std::vector<float>{1.0f}, std::vector<size_t>{1});

        zerograd::batchNorm1d(input, gamma, beta, running_mean, running_var, true, 0.1f);
        
        REQUIRE(running_mean->data[0] == Catch::Approx(0.25f).margin(1e-5f));
        REQUIRE(running_var->data[0] == Catch::Approx(1.066666f).margin(1e-5f));
    }

    SECTION("Wrong shape throws") {
        auto input = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f},
            std::vector<size_t>{3}
        );
        auto gamma = std::make_shared<zerograd::Tensor>(std::vector<float>{1.0f}, std::vector<size_t>{1});
        auto beta = std::make_shared<zerograd::Tensor>(std::vector<float>{0.0f}, std::vector<size_t>{1});
        auto running_mean = std::make_shared<zerograd::Tensor>(std::vector<float>{0.0f}, std::vector<size_t>{1});
        auto running_var = std::make_shared<zerograd::Tensor>(std::vector<float>{1.0f}, std::vector<size_t>{1});

        REQUIRE_THROWS_AS(
            zerograd::batchNorm1d(input, gamma, beta, running_mean, running_var, true),
            std::runtime_error
        );
    }
}

TEST_CASE("BatchNorm1d Backward - Gradient Checking", "[tensor][backward][batchnorm][gradcheck]") {

    SECTION("Gradient check - input") {
        auto input = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
            std::vector<size_t>{3, 2}, true
        );
        auto gamma = std::make_shared<zerograd::Tensor>(std::vector<float>{1.5f, 0.8f}, std::vector<size_t>{2}, true);
        auto beta = std::make_shared<zerograd::Tensor>(std::vector<float>{0.5f, -0.2f}, std::vector<size_t>{2}, true);
        auto running_mean = std::make_shared<zerograd::Tensor>(std::vector<float>{0.0f, 0.0f}, std::vector<size_t>{2});
        auto running_var = std::make_shared<zerograd::Tensor>(std::vector<float>{1.0f, 1.0f}, std::vector<size_t>{2});

        auto forward = [&]() {
            return sum(zerograd::batchNorm1d(input, gamma, beta, running_mean, running_var, true));
        };

        auto out = forward();
        out->backward();

        for (size_t i = 0; i < input->data.size(); ++i) {
            float numerical = compute_tensor_numerical_gradient(forward, input, i);
            REQUIRE(input->grad[i] == Catch::Approx(numerical).margin(1e-2f));
        }
    }

    SECTION("Gradient check - gamma") {
        auto input = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
            std::vector<size_t>{3, 2}, true
        );
        auto gamma = std::make_shared<zerograd::Tensor>(std::vector<float>{1.5f, 0.8f}, std::vector<size_t>{2}, true);
        auto beta = std::make_shared<zerograd::Tensor>(std::vector<float>{0.5f, -0.2f}, std::vector<size_t>{2}, true);
        auto running_mean = std::make_shared<zerograd::Tensor>(std::vector<float>{0.0f, 0.0f}, std::vector<size_t>{2});
        auto running_var = std::make_shared<zerograd::Tensor>(std::vector<float>{1.0f, 1.0f}, std::vector<size_t>{2});

        auto forward = [&]() {
            return sum(zerograd::batchNorm1d(input, gamma, beta, running_mean, running_var, true));
        };

        auto out = forward();
        out->backward();

        for (size_t i = 0; i < gamma->data.size(); ++i) {
            float numerical = compute_tensor_numerical_gradient(forward, gamma, i);
            REQUIRE(gamma->grad[i] == Catch::Approx(numerical).margin(1e-2f));
        }
    }

    SECTION("Gradient check - beta") {
        auto input = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
            std::vector<size_t>{3, 2}, true
        );
        auto gamma = std::make_shared<zerograd::Tensor>(std::vector<float>{1.5f, 0.8f}, std::vector<size_t>{2}, true);
        auto beta = std::make_shared<zerograd::Tensor>(std::vector<float>{0.5f, -0.2f}, std::vector<size_t>{2}, true);
        auto running_mean = std::make_shared<zerograd::Tensor>(std::vector<float>{0.0f, 0.0f}, std::vector<size_t>{2});
        auto running_var = std::make_shared<zerograd::Tensor>(std::vector<float>{1.0f, 1.0f}, std::vector<size_t>{2});

        auto forward = [&]() {
            return sum(zerograd::batchNorm1d(input, gamma, beta, running_mean, running_var, true));
        };

        auto out = forward();
        out->backward();

        for (size_t i = 0; i < beta->data.size(); ++i) {
            float numerical = compute_tensor_numerical_gradient(forward, beta, i);
            REQUIRE(beta->grad[i] == Catch::Approx(numerical).margin(1e-2f));
        }
    }

    SECTION("Gradient check - with non-trivial downstream weighting") {
        auto input = std::make_shared<zerograd::Tensor>(
            std::vector<float>{0.5f, -1.0f, 2.0f, 1.5f, -0.5f, 3.0f},
            std::vector<size_t>{3, 2}, true
        );
        auto gamma = std::make_shared<zerograd::Tensor>(std::vector<float>{1.0f, 1.0f}, std::vector<size_t>{2}, true);
        auto beta = std::make_shared<zerograd::Tensor>(std::vector<float>{0.0f, 0.0f}, std::vector<size_t>{2}, true);
        auto running_mean = std::make_shared<zerograd::Tensor>(std::vector<float>{0.0f, 0.0f}, std::vector<size_t>{2});
        auto running_var = std::make_shared<zerograd::Tensor>(std::vector<float>{1.0f, 1.0f}, std::vector<size_t>{2});
        auto weights = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f, 2.0f, -1.0f, 0.5f, 3.0f, -2.0f},
            std::vector<size_t>{3, 2}
        );

        auto forward = [&]() {
            return sum(zerograd::batchNorm1d(input, gamma, beta, running_mean, running_var, true) * weights);
        };

        auto out = forward();
        out->backward();

        for (size_t i = 0; i < input->data.size(); ++i) {
            float numerical = compute_tensor_numerical_gradient(forward, input, i);
            REQUIRE(input->grad[i] == Catch::Approx(numerical).margin(1e-2f));
        }
    }
}

TEST_CASE("im2col Forward", "[tensor][forward][im2col]") {

    SECTION("Correct output shape") {
        auto img = std::make_shared<zerograd::Tensor>(
            std::vector<float>(1 * 1 * 4 * 4, 1.0f),
            std::vector<size_t>{1, 1, 4, 4}
        );
        auto col = zerograd::im2col(img, 2, 2, 1, 0);
        
        REQUIRE(col->shape == std::vector<size_t>{4, 9});
    }

    SECTION("Known simple 3x3 patch extraction") {
        auto img = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9},
            std::vector<size_t>{1, 1, 3, 3}
        );
        auto col = zerograd::im2col(img, 2, 2, 1, 0);
        
        REQUIRE(col->data[0 * col->shape[1] + 0] == Catch::Approx(1.0f));
        REQUIRE(col->data[1 * col->shape[1] + 0] == Catch::Approx(2.0f));
        REQUIRE(col->data[2 * col->shape[1] + 0] == Catch::Approx(4.0f));
        REQUIRE(col->data[3 * col->shape[1] + 0] == Catch::Approx(5.0f));
    }

    SECTION("Padding produces zero borders") {
        auto img = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1, 1, 1, 1},
            std::vector<size_t>{1, 1, 2, 2}
        );
        auto col = zerograd::im2col(img, 2, 2, 1, 1);
        
        REQUIRE(col->data[0] == Catch::Approx(0.0f));
    }
}

TEST_CASE("conv2d Forward", "[tensor][forward][conv2d]") {

    SECTION("Correct output shape - no padding, stride 1") {
        auto input = std::make_shared<zerograd::Tensor>(
            std::vector<float>(1 * 1 * 5 * 5, 1.0f),
            std::vector<size_t>{1, 1, 5, 5}
        );
        auto weight = std::make_shared<zerograd::Tensor>(
            std::vector<float>(1 * 1 * 3 * 3, 1.0f),
            std::vector<size_t>{1, 1, 3, 3}
        );
        auto result = zerograd::conv2d(input, weight, nullptr, 1, 0);
        REQUIRE(result->shape == std::vector<size_t>{1, 1, 3, 3});
    }

    SECTION("Known values - single channel, identity-like sum kernel") {
        auto input = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9},
            std::vector<size_t>{1, 1, 3, 3}
        );
        auto weight = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1, 0, 0, 0},
            std::vector<size_t>{1, 1, 2, 2}
        );
        auto result = zerograd::conv2d(input, weight, nullptr, 1, 0);
        
        REQUIRE(result->data[0] == Catch::Approx(1.0f));
        REQUIRE(result->data[1] == Catch::Approx(2.0f));
        REQUIRE(result->data[2] == Catch::Approx(4.0f));
        REQUIRE(result->data[3] == Catch::Approx(5.0f));
    }

    SECTION("Bias is added correctly") {
        auto input = std::make_shared<zerograd::Tensor>(
            std::vector<float>(1 * 1 * 3 * 3, 0.0f),
            std::vector<size_t>{1, 1, 3, 3}
        );
        auto weight = std::make_shared<zerograd::Tensor>(
            std::vector<float>(1 * 1 * 2 * 2, 1.0f),
            std::vector<size_t>{1, 1, 2, 2}
        );
        auto bias = std::make_shared<zerograd::Tensor>(
            std::vector<float>{5.0f},
            std::vector<size_t>{1}
        );
        auto result = zerograd::conv2d(input, weight, bias, 1, 0);
        for (float v : result->data)
            REQUIRE(v == Catch::Approx(5.0f));
    }

    SECTION("Multiple output channels") {
        auto input = std::make_shared<zerograd::Tensor>(
            std::vector<float>(1 * 1 * 4 * 4, 1.0f),
            std::vector<size_t>{1, 1, 4, 4}
        );
        auto weight = std::make_shared<zerograd::Tensor>(
            std::vector<float>(2 * 1 * 2 * 2, 1.0f),
            std::vector<size_t>{2, 1, 2, 2}
        );
        auto result = zerograd::conv2d(input, weight, nullptr, 1, 0);
        REQUIRE(result->shape == std::vector<size_t>{1, 2, 3, 3});
    }

    SECTION("Stride > 1 reduces output size correctly") {
        auto input = std::make_shared<zerograd::Tensor>(
            std::vector<float>(1 * 1 * 5 * 5, 1.0f),
            std::vector<size_t>{1, 1, 5, 5}
        );
        auto weight = std::make_shared<zerograd::Tensor>(
            std::vector<float>(1 * 1 * 3 * 3, 1.0f),
            std::vector<size_t>{1, 1, 3, 3}
        );
        auto result = zerograd::conv2d(input, weight, nullptr, 2, 0);
        
        REQUIRE(result->shape == std::vector<size_t>{1, 1, 2, 2});
    }
}

TEST_CASE("conv2d Backward - Gradient Checking", "[tensor][backward][conv2d][gradcheck]") {

    SECTION("Gradient check - input, batch size 1") {
        auto input = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1,2,3,4,5,6,7,8,9},
            std::vector<size_t>{1, 1, 3, 3}, true
        );
        auto weight = std::make_shared<zerograd::Tensor>(
            std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f},
            std::vector<size_t>{1, 1, 2, 2}, true
        );

        auto forward = [&]() { return sum(zerograd::conv2d(input, weight, nullptr, 1, 0)); };

        auto out = forward();
        out->backward();

        for (size_t i = 0; i < input->data.size(); ++i) {
            float numerical = compute_tensor_numerical_gradient(forward, input, i);
            REQUIRE(input->grad[i] == Catch::Approx(numerical).margin(1e-2f));
        }
    }

    SECTION("Gradient check - weight, batch size 1") {
        auto input = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1,2,3,4,5,6,7,8,9},
            std::vector<size_t>{1, 1, 3, 3}, true
        );
        auto weight = std::make_shared<zerograd::Tensor>(
            std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f},
            std::vector<size_t>{1, 1, 2, 2}, true
        );

        auto forward = [&]() { return sum(zerograd::conv2d(input, weight, nullptr, 1, 0)); };

        auto out = forward();
        out->backward();

        for (size_t i = 0; i < weight->data.size(); ++i) {
            float numerical = compute_tensor_numerical_gradient(forward, weight, i);
            REQUIRE(weight->grad[i] == Catch::Approx(numerical).margin(1e-2f));
        }
    }

    SECTION("CRITICAL: Gradient check - input, batch size > 1 (catches layout bugs)") {
        auto input = std::make_shared<zerograd::Tensor>(
            std::vector<float>{
                1,2,3,4,5,6,7,8,9,      
                9,8,7,6,5,4,3,2,1        
            },
            std::vector<size_t>{2, 1, 3, 3}, true
        );
        auto weight = std::make_shared<zerograd::Tensor>(
            std::vector<float>{0.5f, -0.3f, 0.2f, 0.1f},
            std::vector<size_t>{1, 1, 2, 2}, true
        );

        auto forward = [&]() { return sum(zerograd::conv2d(input, weight, nullptr, 1, 0)); };

        auto out = forward();
        out->backward();

        for (size_t i = 0; i < input->data.size(); ++i) {
            float numerical = compute_tensor_numerical_gradient(forward, input, i);
            REQUIRE(input->grad[i] == Catch::Approx(numerical).margin(1e-2f));
        }
    }

    SECTION("CRITICAL: Gradient check - weight, batch size > 1, multiple output channels") {
        auto input = std::make_shared<zerograd::Tensor>(
            std::vector<float>{
                1,2,3,4,5,6,7,8,9,
                0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f
            },
            std::vector<size_t>{2, 1, 3, 3}, true
        );
        auto weight = std::make_shared<zerograd::Tensor>(
            std::vector<float>{0.1f,0.2f,0.3f,0.4f, -0.1f,-0.2f,-0.3f,-0.4f},
            std::vector<size_t>{2, 1, 2, 2}, true
        );

        auto forward = [&]() { return sum(zerograd::conv2d(input, weight, nullptr, 1, 0)); };

        auto out = forward();
        out->backward();

        for (size_t i = 0; i < weight->data.size(); ++i) {
            float numerical = compute_tensor_numerical_gradient(forward, weight, i);
            REQUIRE(weight->grad[i] == Catch::Approx(numerical).margin(1e-2f));
        }
    }

    SECTION("Gradient check - bias") {
        auto input = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1,2,3,4,5,6,7,8,9},
            std::vector<size_t>{1, 1, 3, 3}, true
        );
        auto weight = std::make_shared<zerograd::Tensor>(
            std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f},
            std::vector<size_t>{1, 1, 2, 2}, true
        );
        auto bias = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1.0f},
            std::vector<size_t>{1}, true
        );

        auto forward = [&]() { return sum(zerograd::conv2d(input, weight, bias, 1, 0)); };

        auto out = forward();
        out->backward();

        for (size_t i = 0; i < bias->data.size(); ++i) {
            float numerical = compute_tensor_numerical_gradient(forward, bias, i);
            REQUIRE(bias->grad[i] == Catch::Approx(numerical).margin(1e-2f));
        }
    }

    SECTION("Gradient check - with padding and stride") {
        auto input = std::make_shared<zerograd::Tensor>(
            std::vector<float>{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16},
            std::vector<size_t>{1, 1, 4, 4}, true
        );
        auto weight = std::make_shared<zerograd::Tensor>(
            std::vector<float>{0.2f,-0.1f,0.3f,0.05f,0.4f,-0.2f,0.1f,0.15f,-0.3f},
            std::vector<size_t>{1, 1, 3, 3}, true
        );

        auto forward = [&]() { return sum(zerograd::conv2d(input, weight, nullptr, 2, 1)); };

        auto out = forward();
        out->backward();

        for (size_t i = 0; i < input->data.size(); ++i) {
            float numerical = compute_tensor_numerical_gradient(forward, input, i);
            REQUIRE(input->grad[i] == Catch::Approx(numerical).margin(1e-2f));
        }
    }
}

TEST_CASE("maxPool2d basic values") {
    std::vector<float> data = {
        1,2,3,4,
        5,6,7,8,
        9,10,11,12,
        13,14,15,16
    };
    auto t = std::make_shared<zerograd::Tensor>(data, std::vector<std::size_t>{1,1,4,4}, true);
    auto out = zerograd::maxPool2d(t, 2, 2, 2, 0);
    REQUIRE(out->shape == std::vector<std::size_t>{1,1,2,2});
    REQUIRE(out->data == std::vector<float>{6,8,14,16});
}

TEST_CASE("maxPool2d gradient check") {
    std::vector<float> data = {
        1.0f, 3.2f, 2.1f, 4.4f,
        0.5f, 6.6f, 1.9f, 2.2f,
        3.3f, 1.1f, 5.5f, 0.8f,
        2.7f, 4.1f, 0.3f, 3.9f
    };
    auto t = std::make_shared<zerograd::Tensor>(
        data, std::vector<std::size_t>{1, 1, 4, 4}, true);

    auto compute_graph = [&]() {
        auto pooled = zerograd::maxPool2d(t, 2, 2, 2, 0);
        return zerograd::sum(pooled);
    };

    std::fill(t->grad.begin(), t->grad.end(), 0.0f);
    auto loss = compute_graph();
    loss->backward();

    for (std::size_t i = 0; i < t->data.size(); ++i) {
        float numeric = compute_tensor_numerical_gradient(compute_graph, t, i);
        REQUIRE(numeric == Catch::Approx(t->grad[i]).epsilon(0.01f));
    }
}

TEST_CASE("flatten gradient check") {
    std::vector<float> data = {
        0.2f, -1.3f, 0.7f,
        2.1f, -0.4f, 1.6f,
        0.9f, -2.2f, 1.1f,
        0.3f, 1.8f, -0.6f
    };
    auto t = std::make_shared<zerograd::Tensor>(
        data, std::vector<std::size_t>{1, 3, 2, 2}, true);

    auto compute_graph = [&]() {
        auto flat = zerograd::flatten(t);
        return zerograd::sum(flat);
    };

    std::fill(t->grad.begin(), t->grad.end(), 0.0f);
    auto loss = compute_graph();
    loss->backward();

    for (std::size_t i = 0; i < t->data.size(); ++i) {
        float numeric = compute_tensor_numerical_gradient(compute_graph, t, i);
        REQUIRE(numeric == Catch::Approx(t->grad[i]).epsilon(0.01f));
    }
}