#pragma once

#include <vector>
#include <algorithm>
#include <memory>
#include <iterator>
#include "tensor.h"

namespace zerograd::metrics
{
    inline std::size_t argmax(std::vector<float>::const_iterator begin, std::vector<float>::const_iterator end)
    {
        auto max_it = std::max_element(begin, end);
        return std::distance(begin, max_it);
    }

    inline std::size_t argmax(const std::vector<float>& vec)
    {
        return argmax(vec.begin(), vec.end());
    }

    inline std::size_t count_correct(
        const std::shared_ptr<Tensor>& logits,
        const std::vector<std::size_t>& labels
    )
    {
        std::size_t batch_size = logits->shape[0];
        std::size_t num_classes = logits->shape[1];
        std::size_t correct = 0;

        for (std::size_t b{}; b < batch_size; ++b) {
            auto row_start = logits->data.begin() + (b * num_classes);
            auto row_end = row_start + num_classes;

            std::size_t predicted_class = argmax(row_start, row_end);

            if (predicted_class == labels[b])
                ++correct;
        }

        return correct;
    }

    inline float compute_accuracy(
        const std::shared_ptr<Tensor>& logits,
        const std::vector<std::size_t>& labels
    )
    {
        std::size_t correct = count_correct(logits, labels);
        return (static_cast<float>(correct) / static_cast<float>(labels.size())) * 100.0f;
    }
}