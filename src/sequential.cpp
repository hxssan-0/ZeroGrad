#include <zerograd/sequential.h>
#include <zerograd/tensor.h>

namespace zerograd
{
    std::shared_ptr<Tensor> Sequential::forward(std::shared_ptr<Tensor> input)
    {
        for (auto& layer: layers) {
            input = layer->forward(input);
        }

        return input;
    }

    std::vector<std::shared_ptr<Tensor>> Sequential::parameters()
    {
        std::vector<std::shared_ptr<Tensor>> all_params;

        for (auto& layer: layers) {
            auto layer_params = layer->parameters();
            all_params.insert(all_params.end(), layer_params.begin(), layer_params.end());
        }

        return all_params;
    }

    void Sequential::add(std::shared_ptr<Layer> layer)
    {
        layers.push_back(std::move(layer));
    }
}