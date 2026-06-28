#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <zerograd/scalar.h>
#include <cmath>
#include <functional>

float compute_numerical_gradient(
    const std::function<std::shared_ptr<zerograd::Scalar>()>& compute_graph, 
    const std::shared_ptr<zerograd::Scalar>& target_node, 
    float h = 1e-3f
)
{
    // saving the original data
    float original_data = target_node->data;

    // f(x + h)
    target_node->data = original_data + h;
    auto out_plus = compute_graph();
    float f_x_plus_h = out_plus->data;

    // f(x - h)
    target_node->data = original_data - h;
    auto out_minus = compute_graph();
    float f_x_minus_h = out_minus->data;

    // restoring the original data
    target_node->data = original_data;

    // (f(x + h) - f(x - h)) / (2h)
    return (f_x_plus_h - f_x_minus_h) / (2.0f * h);
}

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

TEST_CASE("Scalar Exponential", "[scalar]") {
    auto a = std::make_shared<zerograd::Scalar>(2.0f);
    auto b = exp(a);

    REQUIRE(b->data == Catch::Approx(std::exp(2.0f)));
    
    REQUIRE(b->get_op() == "exp");
    auto children = b->get_children();
    REQUIRE(children.size() == 1);
    REQUIRE(children[0] == a);
}

TEST_CASE("Scalar Tanh", "[scalar]") {
    auto a = std::make_shared<zerograd::Scalar>(0.5f);
    auto b = tanh(a);

    REQUIRE(b->data == Catch::Approx(std::tanh(0.5f)));
    
    REQUIRE(b->get_op() == "tanh");
    auto children = b->get_children();
    REQUIRE(children.size() == 1);
    REQUIRE(children[0] == a);
}

TEST_CASE("Scalar ReLU", "[scalar]") {
    auto a_pos = std::make_shared<zerograd::Scalar>(3.0f);
    auto b_pos = relu(a_pos);

    REQUIRE(b_pos->data == Catch::Approx(3.0f));
    REQUIRE(b_pos->get_op() == "relu");
    REQUIRE(b_pos->get_children().size() == 1);
    REQUIRE(b_pos->get_children()[0] == a_pos);

    auto a_neg = std::make_shared<zerograd::Scalar>(-5.0f);
    auto b_neg = relu(a_neg);

    REQUIRE(b_neg->data == Catch::Approx(0.0f));
    REQUIRE(b_neg->get_op() == "relu");
    REQUIRE(b_neg->get_children().size() == 1);
    REQUIRE(b_neg->get_children()[0] == a_neg);
    
    auto a_zero = std::make_shared<zerograd::Scalar>(0.0f);
    auto b_zero = relu(a_zero);

    REQUIRE(b_zero->data == Catch::Approx(0.0f));
}

TEST_CASE("Engine Integration - Simple Expression", "[autograd]") {
    auto a = std::make_shared<zerograd::Scalar>(2.0f);
    auto b = std::make_shared<zerograd::Scalar>(-3.0f);
    auto c = std::make_shared<zerograd::Scalar>(10.0f);

    // y = a + b * c
    auto e = b * c;
    auto y = a + e;

    y->backward();

    // forward pass
    REQUIRE(y->data == Catch::Approx(-28.0f));

    // backward pass
    REQUIRE(a->grad == Catch::Approx(1.0f));
    REQUIRE(b->grad == Catch::Approx(10.0f));
    REQUIRE(c->grad == Catch::Approx(-3.0f));
}

TEST_CASE("Engine Integration - A Simple Neuron", "[autograd]") {
    auto x1 = std::make_shared<zerograd::Scalar>(2.0f);
    auto w1 = std::make_shared<zerograd::Scalar>(-3.0f);
    
    auto x2 = std::make_shared<zerograd::Scalar>(0.0f);
    auto w2 = std::make_shared<zerograd::Scalar>(1.0f);
    
    auto b  = std::make_shared<zerograd::Scalar>(6.8813735870195432f);

    // x1*w1 + x2*w2 + b
    auto x1w1 = x1 * w1;
    auto x2w2 = x2 * w2;
    auto x1w1x2w2 = x1w1 + x2w2;
    auto n = x1w1x2w2 + b;
    
    auto o = tanh(n);

    o->backward();

    // forward pass
    REQUIRE(o->data == Catch::Approx(0.7071f).margin(0.01f));

    // backward pass
    REQUIRE(x1->grad == Catch::Approx(-1.5f).margin(0.01f));
    REQUIRE(w1->grad == Catch::Approx(1.0f).margin(0.01f));
    REQUIRE(x2->grad == Catch::Approx(0.5f).margin(0.01f));
    REQUIRE(w2->grad == Catch::Approx(0.0f).margin(0.01f));
}

TEST_CASE("Numerical Gradient Check", "[autograd]") {
    auto a = std::make_shared<zerograd::Scalar>(2.0f);
    auto b = std::make_shared<zerograd::Scalar>(3.0f);

    auto forward = [&a, &b]() {
        return (a * a) + (b * a); // a^2 + ab
    };

    auto out = forward();
    out->backward();
    float analytical_grad_a = a->grad; // the value from my engine's backward pass

    // the value from the numerical gradient check
    float numerical_grad_a = compute_numerical_gradient(forward, a);

    REQUIRE(analytical_grad_a == Catch::Approx(numerical_grad_a).margin(1e-5f));
}

TEST_CASE("Gradient Accumulation - Shared Input", "[autograd]") {
    auto a = std::make_shared<zerograd::Scalar>(3.0f);
    auto b = a + a;

    b->backward();

    REQUIRE(a->grad == Catch::Approx(2.0f));
}

TEST_CASE("Gradient Accumulation - Used In Multiple Ops", "[autograd]") {
    auto a = std::make_shared<zerograd::Scalar>(2.0f);
    auto b = a * a * a;

    b->backward();

    REQUIRE(a->grad == Catch::Approx(12.0f));
}

TEST_CASE("Numerical Gradient - pow", "[autograd]") {
    auto a = std::make_shared<zerograd::Scalar>(2.0f);
    auto forward = [&a]() { return pow(a, 3.0f); };

    auto out = forward();
    out->backward();

    REQUIRE(a->grad == Catch::Approx(compute_numerical_gradient(forward, a)).margin(1e-3f));
}

TEST_CASE("Numerical Gradient - exp", "[autograd]") {
    auto a = std::make_shared<zerograd::Scalar>(1.0f);
    auto forward = [&a]() { return exp(a); };

    auto out = forward();
    out->backward();

    REQUIRE(a->grad == Catch::Approx(compute_numerical_gradient(forward, a)).margin(1e-3f));
}

TEST_CASE("Numerical Gradient - tanh", "[autograd]") {
    auto a = std::make_shared<zerograd::Scalar>(0.5f);
    auto forward = [&a]() { return tanh(a); };

    auto out = forward();
    out->backward();

    REQUIRE(a->grad == Catch::Approx(compute_numerical_gradient(forward, a)).margin(1e-3f));
}

TEST_CASE("Numerical Gradient - relu", "[autograd]") {
    auto a = std::make_shared<zerograd::Scalar>(1.5f);
    auto forward = [&a]() { return relu(a); };

    auto out = forward();
    out->backward();

    REQUIRE(a->grad == Catch::Approx(compute_numerical_gradient(forward, a)).margin(1e-3f));
}

TEST_CASE("Numerical Gradient - division", "[autograd]") {
    auto a = std::make_shared<zerograd::Scalar>(4.0f);
    auto b = std::make_shared<zerograd::Scalar>(2.0f);
    auto forward = [&a, &b]() { return a / b; };

    auto out = forward();
    out->backward();

    REQUIRE(a->grad == Catch::Approx(compute_numerical_gradient(forward, a)).margin(1e-3f));
    a->grad = 0.0f; b->grad = 0.0f;
    auto out2 = forward();
    out2->backward();
    REQUIRE(b->grad == Catch::Approx(compute_numerical_gradient(forward, b)).margin(1e-3f));
}

TEST_CASE("Deep Chain Rule", "[autograd]") {
    auto a = std::make_shared<zerograd::Scalar>(2.0f);

    // y = tanh(exp(a^2))
    auto forward = [&a]() { return tanh(exp(pow(a, 2.0f))); };

    auto out = forward();
    out->backward();

    REQUIRE(a->grad == Catch::Approx(compute_numerical_gradient(forward, a)).margin(1e-3f));
}

TEST_CASE("ReLU at zero - grad is zero", "[autograd]") {
    auto a = std::make_shared<zerograd::Scalar>(0.0f);
    auto b = relu(a);
    b->backward();

    REQUIRE(a->grad == Catch::Approx(0.0f));
}

TEST_CASE("Division by value close to zero - pow backward stability", "[autograd]") {
    auto a = std::make_shared<zerograd::Scalar>(0.1f);
    auto b = std::make_shared<zerograd::Scalar>(0.1f);
    auto forward = [&a, &b]() { return a / b; };

    auto out = forward();
    out->backward();

    REQUIRE(std::isfinite(a->grad));
    REQUIRE(std::isfinite(b->grad));
}

TEST_CASE("Backward only called once - grad not double accumulated", "[autograd]") {
    auto a = std::make_shared<zerograd::Scalar>(2.0f);
    auto b = std::make_shared<zerograd::Scalar>(3.0f);
    auto c = a * b;

    c->backward();
    REQUIRE(a->grad == Catch::Approx(3.0f));
}