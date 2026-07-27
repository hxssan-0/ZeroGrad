#include <catch2/catch_test_macros.hpp>
#include <zerograd/data/mnist.h>
#include <iostream>

TEST_CASE("MNIST Loading", "[mnist][data]") {

    SECTION("Loads correct number of samples") {
        auto data = zerograd::load_mnist(
            "../datasets/mnist/train_images.idx3-ubyte",
            "../datasets/mnist/train_labels.idx1-ubyte"
        );
        REQUIRE(data.num_samples == 60000);
        REQUIRE(data.image_size == 784);
    }

    SECTION("Pixel values are normalized to [0, 1]") {
        auto data = zerograd::load_mnist(
            "../datasets/mnist/train_images.idx3-ubyte",
            "../datasets/mnist/train_labels.idx1-ubyte"
        );
        for (float pixel : data.images) {
            REQUIRE(pixel >= 0.0f);
            REQUIRE(pixel <= 1.0f);
        }
    }

    SECTION("Labels are valid digit values") {
        auto data = zerograd::load_mnist(
            "../datasets/mnist/train_images.idx3-ubyte",
            "../datasets/mnist/train_labels.idx1-ubyte"
        );
        for (uint8_t label : data.labels) {
            REQUIRE(label <= 9);
        }
    }

    SECTION("Throws on missing file") {
        REQUIRE_THROWS_AS(
            zerograd::load_mnist("nonexistent.idx", "also_nonexistent.idx"),
            std::runtime_error
        );
    }

    SECTION("Visual spot check - print first 3 digits") {
        auto data = zerograd::load_mnist(
            "../datasets/mnist/train_images.idx3-ubyte",
            "../datasets/mnist/train_labels.idx1-ubyte"
        );

        for (int i = 0; i < 3; ++i) {
            std::size_t offset = i * data.image_size;
            std::cout << "Label: " << static_cast<int>(data.labels[i]) << "\n";
            zerograd::print_digit_ascii(data.images, offset);
        }

        SUCCEED("Visually inspect printed digits above");
    }
}