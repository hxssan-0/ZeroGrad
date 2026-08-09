#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <zerograd/data/mnist.hpp>
#include <numeric>

TEST_CASE("DataLoader", "[dataloader][data]") {
    auto make_fake_data = [](std::size_t num_samples, std::size_t image_size) {
        std::vector<float> images(num_samples * image_size);
        std::vector<uint8_t> labels(num_samples);

        for (std::size_t i = 0; i < num_samples; ++i) {
            for (std::size_t j = 0; j < image_size; ++j) {
                images[i * image_size + j] = static_cast<float>(i);
            }
            labels[i] = static_cast<uint8_t>(i % 10);
        }

        return zerograd::MNISTData(images, labels, num_samples, image_size);
    };

    SECTION("Constructor throws if batch_size > num_samples") {
        auto data = make_fake_data(5, 4);
        REQUIRE_THROWS_AS(zerograd::DataLoader(data, 10), std::invalid_argument);
    }

    SECTION("has_next() true initially, false after exhausting full batches") {
        auto data = make_fake_data(10, 4);
        zerograd::DataLoader loader(data, 3);

        REQUIRE(loader.has_next());

        loader.next_batch();
        loader.next_batch();
        loader.next_batch();

        REQUIRE_FALSE(loader.has_next());
    }

    SECTION("next_batch() returns correct shape") {
        auto data = make_fake_data(10, 4);
        zerograd::DataLoader loader(data, 3);

        auto [images, labels] = loader.next_batch();

        REQUIRE(images->shape == std::vector<std::size_t>{3, 4});
        REQUIRE(labels.size() == 3);
    }

    SECTION("next_batch() without shuffle returns sequential samples") {
        auto data = make_fake_data(6, 4);
        zerograd::DataLoader loader(data, 2);

        auto [images, labels] = loader.next_batch();

        REQUIRE(images->data[0] == Catch::Approx(0.0f));
        REQUIRE(images->data[4] == Catch::Approx(1.0f));
    }

    SECTION("current_pos advances correctly across batches") {
        auto data = make_fake_data(9, 4);
        zerograd::DataLoader loader(data, 3);

        auto [images1, labels1] = loader.next_batch();
        auto [images2, labels2] = loader.next_batch();

        REQUIRE(images2->data[0] == Catch::Approx(3.0f));
    }

    SECTION("reset() allows re-iterating from the start") {
        auto data = make_fake_data(6, 4);
        zerograd::DataLoader loader(data, 2);

        loader.next_batch();
        loader.next_batch();
        loader.next_batch();
        REQUIRE_FALSE(loader.has_next());

        loader.reset();
        REQUIRE(loader.has_next());

        auto [images, labels] = loader.next_batch();
        REQUIRE(images->data[0] == Catch::Approx(0.0f));
    }

    SECTION("shuffle() changes the order of batches") {
        auto data = make_fake_data(20, 4);
        zerograd::DataLoader loader(data, 4);

        auto [images_before, labels_before] = loader.next_batch();
        loader.reset();

        loader.shuffle();
        auto [images_after, labels_after] = loader.next_batch();

        bool identical = true;
        for (std::size_t i = 0; i < images_before->data.size(); ++i) {
            if (images_before->data[i] != images_after->data[i]) {
                identical = false;
                break;
            }
        }
        REQUIRE_FALSE(identical);
    }

    SECTION("Labels match the correct sample after shuffling") {
        auto data = make_fake_data(10, 4);
        zerograd::DataLoader loader(data, 5);
        loader.shuffle();

        auto [images, labels] = loader.next_batch();

        for (std::size_t i = 0; i < labels.size(); ++i) {
            float pixel_value = images->data[i * 4];
            std::size_t expected_label = static_cast<std::size_t>(pixel_value) % 10;
            REQUIRE(labels[i] == expected_label);
        }
    }
}