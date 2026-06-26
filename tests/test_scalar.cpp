#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <zerograd/scalar.h>

TEST_CASE("Scalar Initialization", "[scalar]") {
    auto a = std::make_shared<zerograd::Scalar>(42.0f);
    REQUIRE(a->data == Catch::Approx(42.0f));
    REQUIRE(a->grad == Catch::Approx(0.0f));
}

TEST_CASE("Scalar Addition - 1", "[scalar]") {
    auto a = std::make_shared<zerograd::Scalar>(3.0f);
    auto b = std::make_shared<zerograd::Scalar>(4.0f);
    auto c = a + b;
    
    REQUIRE(c->data == Catch::Approx(7.0f));
    REQUIRE(c->grad == Catch::Approx(0.0f));
    
    REQUIRE(c->get_op() == "+");
    
    auto children = c->get_children();
    REQUIRE(children.size() == 2);
    REQUIRE(children[0] == a);
    REQUIRE(children[1] == b);
}

TEST_CASE("Scalar Addition - 2", "[scalar]") {
    auto a = std::make_shared<zerograd::Scalar>(3.0f);
    auto b = a + 4.0f;
    
    REQUIRE(b->data == Catch::Approx(7.0f));
    REQUIRE(b->grad == Catch::Approx(0.0f));
    
    REQUIRE(b->get_op() == "+");
    auto children = b->get_children();
    REQUIRE(children.size() == 2);
    REQUIRE(children[0] == a);
    REQUIRE(children[1]->data == Catch::Approx(4.0f));
}

TEST_CASE("Scalar Addition - 3", "[scalar]") {
    auto a = std::make_shared<zerograd::Scalar>(3.0f);
    auto b = 4.0f + a;
    
    REQUIRE(b->data == Catch::Approx(7.0f));
    REQUIRE(b->grad == Catch::Approx(0.0f));
    
    REQUIRE(b->get_op() == "+");
    auto children = b->get_children();
    REQUIRE(children.size() == 2);
    REQUIRE(children[0]->data == Catch::Approx(4.0f));
    REQUIRE(children[1] == a);
}

TEST_CASE("Scalar Multiplication - 1", "[scalar]") {
    auto a = std::make_shared<zerograd::Scalar>(3.0f);
    auto b = std::make_shared<zerograd::Scalar>(4.0f);
    auto c = a * b;
    
    REQUIRE(c->data == Catch::Approx(12.0f));
    REQUIRE(c->grad == Catch::Approx(0.0f));
    
    REQUIRE(c->get_op() == "*");
    auto children = c->get_children();
    REQUIRE(children.size() == 2);
    REQUIRE(children[0] == a);
    REQUIRE(children[1] == b);
}

TEST_CASE("Scalar Multiplication - 2", "[scalar]") {
    auto a = std::make_shared<zerograd::Scalar>(3.0f);
    auto b = a * 4.0f;
    
    REQUIRE(b->data == Catch::Approx(12.0f));
    REQUIRE(b->grad == Catch::Approx(0.0f));
    
    REQUIRE(b->get_op() == "*");
    auto children = b->get_children();
    REQUIRE(children.size() == 2);
    REQUIRE(children[0] == a);
    REQUIRE(children[1]->data == Catch::Approx(4.0f));
}

TEST_CASE("Scalar Multiplication - 3", "[scalar]") {
    auto a = std::make_shared<zerograd::Scalar>(3.0f);
    auto b = 4.0f * a;
    
    REQUIRE(b->data == Catch::Approx(12.0f));
    REQUIRE(b->grad == Catch::Approx(0.0f));
    
    REQUIRE(b->get_op() == "*");
    auto children = b->get_children();
    REQUIRE(children.size() == 2);
    REQUIRE(children[0]->data == Catch::Approx(4.0f));
    REQUIRE(children[1] == a);
}

TEST_CASE("Backward Pass - Manual", "[scalar]") {
    auto a = std::make_shared<zerograd::Scalar>(2.0f);
    auto b = std::make_shared<zerograd::Scalar>(3.0f);
    auto c = a * b;

    c->grad = 1.0f;

    c->call_backward();

    REQUIRE(a->grad == Catch::Approx(3.0f));
    REQUIRE(b->grad == Catch::Approx(2.0f));
}