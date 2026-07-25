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
        std::vector<float> images; // flattened and normalize in the 0-1 range
        std::vector<uint8_t> labels;
        std::size_t num_samples;
    };

    MNISTData load_mnist(const std::string& images_path, const std::string& labels_path);

    class DataLoader
    {
    private:
        MNISTData data;
        std::vector<std::size_t> indices;
        std::size_t batch_size;
        std::size_t current_pos;

    public:
        DataLoader(MNISTData data, std::size_t batch_size);
        void shuffle();
        bool has_next();
        std::pair<std::shared_ptr<Tensor>, std::vector<std::size_t>> next_batch();
        void reset();
    };
}