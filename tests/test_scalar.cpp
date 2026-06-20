#include <catch2/catch_test_macros.hpp>
#include <zerograd/scalar.h>

TEST_CASE("Scalar Initialization", "[scalar]") {
    zerograd::Scalar a(42.0f);
    REQUIRE(a.data == 42.0f);
}