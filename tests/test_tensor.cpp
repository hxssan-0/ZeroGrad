#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <zerograd/tensor.h>
#include <vector>
#include <memory>

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
}