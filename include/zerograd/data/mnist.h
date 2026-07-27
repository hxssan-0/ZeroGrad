#pragma once

#include "../tensor.h"
#include <cstdint>
#include <vector>
#include <cstddef>
#include <string>
#include <utility>
#include <memory>

namespace zerograd
{

    struct MNISTData
    {
        std::vector<float> images; // flattened and normalized in the 0-1 range
        std::vector<uint8_t> labels;
        std::size_t num_samples;
        std::size_t image_size;

        MNISTData(std::vector<float> images, std::vector<uint8_t> labels, std::size_t num_samples, std::size_t image_size);
    };

    MNISTData load_mnist(const std::string& images_path, const std::string& labels_path);
    uint32_t reverse_bytes(uint32_t value);
    void print_digit_ascii(const std::vector<float>& image, std::size_t offset);
    std::vector<float> normalize_data(std::vector<uint8_t> data);

    class DataLoader
    {
    private:
        MNISTData data;
        std::vector<std::size_t> indices; // will hold indices 0, 1, 2, ..., num_samples-1 then shuffled
        std::size_t batch_size;
        std::size_t current_pos;

    public:
        DataLoader(MNISTData data, std::size_t batch_size);
        void shuffle();
        bool has_next();
        // takes the next batch_size indices, gathers the corresponding images/labels
        // i.e. returns a Tensor of shape (batch_size, 784) and a vector of the labels
        std::pair<std::shared_ptr<Tensor>, std::vector<std::size_t>> next_batch();
        void reset(); // for starting a new epoch
    };
}