#include <zerograd/data/mnist.h>
#include <iostream>

int main()
{
    auto data = zerograd::load_mnist(
        "../datasets/mnist/train_images.idx3-ubyte",
        "../datasets/mnist/train_labels.idx1-ubyte"
    );

    std::cout << "Loaded " << data.num_samples << " samples\n\n";

    // print first 5 digits with their labels
    for (int i = 0; i < 5; ++i) {
        std::size_t offset = i * data.image_size;
        int label = static_cast<int>(data.labels[i]);

        std::cout << "Sample " << i << " - Label: " << label << "\n";
        zerograd::print_digit_ascii(data.images, offset);
        std::cout << "\n";
    }

    return 0;
}