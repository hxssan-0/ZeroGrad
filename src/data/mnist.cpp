#include <zerograd/data/mnist.h>
#include <fstream>
#include <iostream>

#define MAGIC_NUMBER_BYTES 4

namespace zerograd
{
    uint32_t reverse_bytes(uint32_t value)
    {
        uint32_t leftmost_byte = value & 0x000000FF;
        uint32_t left_middle_byte = (value & 0x0000FF00) >> 8;
        uint32_t right_middle_byte = (value & 0x00FF0000) >> 16;
        uint32_t rightmost_byte = (value & 0xFF000000) >> 24;

        leftmost_byte <<= 24;
        left_middle_byte <<= 16;
        right_middle_byte <<= 8;

        uint32_t reversed_value = leftmost_byte | left_middle_byte | right_middle_byte | rightmost_byte;
        return reversed_value;
    }

    void print_digit_ascii(const std::vector<float>& image, std::size_t offset) {
        const std::string chars = " .:-=+*#%@";
        for (int row = 0; row < 28; ++row) {
            for (int col = 0; col < 28; ++col) {
                float pixel = image[offset + row*28 + col];
                std::cout << chars[static_cast<int>(pixel * (chars.size()-1))];
            }
            std::cout << '\n';
        }
    }

    std::vector<float> normalize_data(std::vector<uint8_t> data)
    {
        std::vector<float> normalized_data(data.size());

        for (std::size_t i{}; i < normalized_data.size(); ++i) {
            normalized_data[i] = static_cast<float>(data[i]) / 255.0f;
        }

        return normalized_data;
    }

    MNISTData load_mnist(const std::string& images_path, const std::string& labels_path)
    {
        std::ifstream images_file(images_path, std::ios::binary);
        std::ifstream labels_file(labels_path, std::ios::binary);

        if (!images_file.is_open()) {
            throw std::runtime_error("Error opening images file\n");
        }

        if (!labels_file.is_open()) {
            throw std::runtime_error("Error opening labels file\n");
        }

        uint8_t images_magic_number[MAGIC_NUMBER_BYTES];
        images_file.read(reinterpret_cast<char*>(images_magic_number), MAGIC_NUMBER_BYTES);

        uint8_t labels_magic_number[MAGIC_NUMBER_BYTES];
        labels_file.read(reinterpret_cast<char*>(labels_magic_number), MAGIC_NUMBER_BYTES);

        int images_data_type = static_cast<int>(images_magic_number[2]);
        int images_num_dimensions = static_cast<int>(images_magic_number[3]);
        int labels_data_type = static_cast<int>(labels_magic_number[2]);
        int labels_num_dimensions = static_cast<int>(labels_magic_number[3]);

        if (images_magic_number[0] != 0 || images_magic_number[1] != 0 || images_data_type != 0x08 || images_num_dimensions != 3) {
            throw std::runtime_error("Invalid images IDX file format\n");
        }

        if (labels_magic_number[0] != 0 || labels_magic_number[1] != 0 || labels_data_type != 0x08 || labels_num_dimensions != 1) {
            throw std::runtime_error("Invalid labels IDX file format\n");
        }

        std::cout << "Images data type code: 0x" << std::hex << images_data_type << std::dec << '\n';
        std::cout << "Images number of dimensions: " << images_num_dimensions << '\n';
        std::cout << "Labels data type code: 0x" << std::hex << labels_data_type << std::dec << '\n';
        std::cout << "Labels number of dimensions: " << labels_num_dimensions << '\n';

        uint32_t images_num_samples;
        images_file.read(reinterpret_cast<char*>(&images_num_samples), 4);
        uint32_t labels_num_samples;
        labels_file.read(reinterpret_cast<char*>(&labels_num_samples), 4);

        images_num_samples = reverse_bytes(images_num_samples);
        labels_num_samples = reverse_bytes(labels_num_samples);

        if (images_num_samples != labels_num_samples) {
            throw std::runtime_error("Unequal number of samples in image and label files\n");
        }

        std::cout << "Number of samples: " << images_num_samples << '\n';

        uint32_t image_rows, image_cols;
        images_file.read(reinterpret_cast<char*>(&image_rows), 4);
        images_file.read(reinterpret_cast<char*>(&image_cols), 4);
        image_rows = reverse_bytes(image_rows);
        image_cols = reverse_bytes(image_cols);
        uint32_t image_size = image_rows * image_cols;

        if (image_size != 784) {
            throw std::runtime_error("Wrong values of images_rows and/or image_cols\n");
        }

        std::cout << "Image size: " << image_size << '\n';

        // reading the image pixel values
        std::vector<uint8_t> data_buffer(images_num_samples * image_size);
        images_file.read(reinterpret_cast<char*>(data_buffer.data()), images_num_samples * image_size);
        std::cout << "Successfully read " << data_buffer.size() << " worth of image data\n";

        // reading the label values
        std::vector<uint8_t> labels_buffer(labels_num_samples);
        labels_file.read(reinterpret_cast<char*>(labels_buffer.data()), labels_num_samples);
        std::cout << "Successfully read " << labels_buffer.size() << " worth of label data\n";

        // preparing the MNISTData struct members
        std::size_t m_num_samples = static_cast<std::size_t>(images_num_samples);
        std::size_t m_image_size = static_cast<std::size_t>(image_size);
        std::vector<float> m_images = normalize_data(data_buffer);

        MNISTData mnist_data(m_images, labels_buffer, m_num_samples, m_image_size);

        images_file.close();
        labels_file.close();

        return mnist_data;
    }

    MNISTData::MNISTData(std::vector<float> images, std::vector<uint8_t> labels, std::size_t num_samples, std::size_t image_size)
        : images(std::move(images)), labels(std::move(labels)), num_samples(num_samples), image_size(image_size)
        {
        }
}