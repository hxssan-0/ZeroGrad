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

TEST_CASE("Scalar Negation", "[scalar]") {
    auto a = std::make_shared<zerograd::Scalar>(5.0f);
    auto b = -a;

    REQUIRE(b->data == Catch::Approx(-5.0f));
    REQUIRE(b->grad == Catch::Approx(0.0f));

    REQUIRE(b->get_op() == "*");
    auto children = b->get_children();
    REQUIRE(children.size() == 2);
    REQUIRE(children[0] == a);
    REQUIRE(children[1]->data == Catch::Approx(-1.0f));
}

TEST_CASE("Scalar Subtraction - 1", "[scalar]") {
    auto a = std::make_shared<zerograd::Scalar>(5.0f);
    auto b = std::make_shared<zerograd::Scalar>(3.0f);
    auto c = a - b;

    REQUIRE(c->data == Catch::Approx(2.0f));
    
    REQUIRE(c->get_op() == "+");
    auto children = c->get_children();
    REQUIRE(children.size() == 2);
    REQUIRE(children[0] == a);
    REQUIRE(children[1]->get_op() == "*");
    REQUIRE(children[1]->data == Catch::Approx(-3.0f));
}

TEST_CASE("Scalar Subtraction - 2", "[scalar]") {
    auto a = std::make_shared<zerograd::Scalar>(5.0f);
    auto b = a - 3.0f;

    REQUIRE(b->data == Catch::Approx(2.0f));
    REQUIRE(b->get_op() == "+");
    REQUIRE(b->get_children()[0] == a);
    REQUIRE(b->get_children()[1]->data == Catch::Approx(-3.0f));
}

TEST_CASE("Scalar Power", "[scalar]") {
    auto a = std::make_shared<zerograd::Scalar>(3.0f);
    auto b = pow(a, 2.0f);

    REQUIRE(b->data == Catch::Approx(9.0f));
    
    REQUIRE(b->get_op() == "^2.000000"); 
    auto children = b->get_children();
    REQUIRE(children.size() == 1);
    REQUIRE(children[0] == a);
}

TEST_CASE("Scalar Division - 1", "[scalar]") {
    auto a = std::make_shared<zerograd::Scalar>(10.0f);
    auto b = std::make_shared<zerograd::Scalar>(2.0f);
    auto c = a / b;

    REQUIRE(c->data == Catch::Approx(5.0f));

    REQUIRE(c->get_op() == "*");
    auto children = c->get_children();
    REQUIRE(children.size() == 2);
    REQUIRE(children[0] == a);
    REQUIRE(children[1]->get_op() == "^-1.000000");
    REQUIRE(children[1]->data == Catch::Approx(0.5f));
}

TEST_CASE("Scalar Division - 2", "[scalar]") {
    auto a = std::make_shared<zerograd::Scalar>(10.0f);
    auto b = a / 2.0f;

    REQUIRE(b->data == Catch::Approx(5.0f));

    REQUIRE(b->get_op() == "*");
    auto children = b->get_children();
    REQUIRE(children.size() == 2);
    REQUIRE(children[0] == a);
    REQUIRE(children[1]->data == Catch::Approx(0.5f));
}

TEST_CASE("Scalar Division - 3", "[scalar]") {
    auto a = std::make_shared<zerograd::Scalar>(2.0f);
    auto b = 10.0f / a;

    REQUIRE(b->data == Catch::Approx(5.0f));

    REQUIRE(b->get_op() == "*");
    auto children = b->get_children();
    REQUIRE(children.size() == 2);
    REQUIRE(children[0]->data == Catch::Approx(10.0f));
    REQUIRE(children[1]->get_op() == "^-1.000000");
}