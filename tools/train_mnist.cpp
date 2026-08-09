#include <zerograd/data/mnist.hpp>
#include <zerograd/metrics.hpp>
#include <zerograd/sequential.hpp>
#include <zerograd/linear.hpp>
#include <zerograd/activations.hpp>
#include <zerograd/optimizer.hpp>
#include <iostream>

int main()
{
    auto train_data = zerograd::load_mnist(
        "../datasets/mnist/train_images.idx3-ubyte",
        "../datasets/mnist/train_labels.idx1-ubyte"
    );

    auto test_data = zerograd::load_mnist(
        "../datasets/mnist/test_images.idx3-ubyte",
        "../datasets/mnist/test_labels.idx1-ubyte"
    );

    zerograd::Sequential mlp;
    mlp.add(std::make_shared<zerograd::Linear>(784, 128));
    mlp.add(std::make_shared<zerograd::ReLU>());
    mlp.add(std::make_shared<zerograd::Linear>(128, 10));

    std::size_t batch_size = 64;
    zerograd::DataLoader train_loader(train_data, batch_size);

    float lr = 1e-1;
    zerograd::Optimizer optimizer(mlp.parameters(), lr);

    // TRAINING LOOP
    std::cout << "====== BEGINNING TRAINING LOOP ======\n";
    constexpr int NUM_EPOCHS = 15;
    for (int epoch{}; epoch < NUM_EPOCHS; ++epoch) {
        float running_loss{};
        std::cout << "========= EPOCH #" << epoch+1 << " =============\n"; 
        train_loader.shuffle();
        train_loader.reset();
        int num_batches = 0;
        while (train_loader.has_next()) {
            std::pair<std::shared_ptr<zerograd::Tensor>, std::vector<std::size_t>> batch = train_loader.next_batch();
            optimizer.zero_grad();
            std::shared_ptr<zerograd::Tensor> logits = mlp.forward(batch.first);
            std::shared_ptr<zerograd::Tensor> loss = ce_loss(logits, batch.second);
            loss->backward();
            optimizer.step();
            running_loss += loss->data[0];

            ++num_batches;

            if ((num_batches % 100) == 0) {
                std::cout << "Batch #" << num_batches << " loss: " << loss->data[0] << '\n';
            }
        }
        std::cout << "Avg. loss: " << running_loss / (num_batches) << '\n';
        std::cout << "================================================\n";
    }

    // TESTING
    std::cout << "====== BEGINNING TESTING ======\n";
    
    zerograd::DataLoader test_loader(test_data, batch_size);
    std::size_t total_correct = 0;
    std::size_t total_samples = 0;

    while (test_loader.has_next()) {
        std::pair<std::shared_ptr<zerograd::Tensor>, std::vector<std::size_t>> batch = test_loader.next_batch();
        std::shared_ptr<zerograd::Tensor> logits = mlp.forward(batch.first);
        total_correct += zerograd::metrics::count_correct(logits, batch.second);
        total_samples += batch.second.size();
    }

    float test_accuracy = (static_cast<float>(total_correct) / static_cast<float>(total_samples)) * 100.0f;
    
    std::cout << "Test Accuracy: " << test_accuracy << "% (" 
              << total_correct << "/" << total_samples << ")\n";

    std::cout << "=================================\n";

    return 0;
}