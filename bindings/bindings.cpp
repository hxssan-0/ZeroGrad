#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <zerograd/tensor.hpp>
#include <zerograd/layer.hpp>
#include <zerograd/linear.hpp>
#include <zerograd/sequential.hpp>
#include <zerograd/optimizer.hpp>
#include <zerograd/activations.hpp>
#include <zerograd/data/mnist.hpp>
#include <zerograd/metrics.hpp>
#include <zerograd/hidden.hpp>
#include <zerograd/graph_analyzer.hpp>
#include <zerograd/arena.hpp>

namespace py = pybind11;

PYBIND11_MODULE(_zerograd_backend, m)
{
    m.doc() = "ZeroGrad C++ Backend";

    // Arena class
    py::class_<zerograd::Arena>(m, "Arena")
        .def(py::init<std::size_t>(), py::arg("total_bytes"))
        .def("reset", &zerograd::Arena::reset);

    // Tensor class
    py::class_<zerograd::Tensor, std::shared_ptr<zerograd::Tensor>>(m, "Tensor")
        .def(py::init<
                std::vector<float>,
                std::vector<std::size_t>,
                bool,
                std::vector<std::shared_ptr<zerograd::Tensor>>,
                std::string,
                zerograd::Arena*,
                zerograd::Arena*
                >(),
            py::arg("data"),
            py::arg("shape"),
            py::arg("requires_grad") = false,
            py::arg("children") = std::vector<std::shared_ptr<zerograd::Tensor>>{},
            py::arg("op") = "",
            py::arg("data_arena") = static_cast<zerograd::Arena*>(nullptr),
            py::arg("grad_arena") = static_cast<zerograd::Arena*>(nullptr)
        )

        .def_property("data", [](const zerograd::Tensor& t) { return std::vector<float>(t.data.begin(), t.data.end()); },
            [](zerograd::Tensor& t, const std::vector<float>& v) { std::copy(v.begin(), v.end(), t.data.begin()); })
        .def_property("shape",
            [](const zerograd::Tensor& t) { 
                return std::vector<std::size_t>(t.shape.begin(), t.shape.end()); 
            },
            [](zerograd::Tensor& t, const std::vector<std::size_t>& new_shape) {
                t.shape = zerograd::ShapeStorage(new_shape);
            }
        )
        .def_property_readonly("strides",
            [](const zerograd::Tensor& t) { return std::vector<std::size_t>(t.strides.begin(), t.strides.end()); })
        .def_property("grad",
            [](const zerograd::Tensor& t) { return std::vector<float>(t.grad.begin(), t.grad.end()); },
            [](zerograd::Tensor& t, const std::vector<float>& v) { std::copy(v.begin(), v.end(), t.grad.begin()); })
        .def_readonly("op", &zerograd::Tensor::_op)
        .def_readonly("birth_step", &zerograd::Tensor::birth_step)

        .def("requires_grad", &zerograd::Tensor::get_requires_grad)

        .def("backward", &zerograd::Tensor::backward)

        .def("__add__", [](const std::shared_ptr<zerograd::Tensor>& a, const std::shared_ptr<zerograd::Tensor>& b) {
            return a + b;
        })
        .def("__sub__", [](const std::shared_ptr<zerograd::Tensor>& a, const std::shared_ptr<zerograd::Tensor>& b) {
            return a - b;
        })
        .def("__mul__", [](const std::shared_ptr<zerograd::Tensor>& a, const std::shared_ptr<zerograd::Tensor>& b) {
            return a * b;
        })

        .def("__repr__", [](const std::shared_ptr<zerograd::Tensor>& t) {
            std::string s = "Tensor(shape=[";
            for (std::size_t i = 0; i < t->shape.size(); ++i) {
                s += std::to_string(t->shape[i]);
                if (i + 1 < t->shape.size()) s += ", ";
            }
            s += "])";
            return s;
        });

        m.def("add", &zerograd::add);
        m.def("sub", &zerograd::sub);
        m.def("mul", &zerograd::mul);
        m.def("matmul", &zerograd::matmul);
        m.def("log", &zerograd::log);
        m.def("exp", &zerograd::exp);

        m.def("relu", &zerograd::relu);
        m.def("sigmoid", &zerograd::sigmoid);
        m.def("tanh", &zerograd::tanh);

        m.def("sum", &zerograd::sum);
        m.def("mean", &zerograd::mean);
        m.def("max", &zerograd::max);

        m.def("transpose", &zerograd::transpose);

        m.def("batchNorm1d", &zerograd::batchNorm1d);
        m.def("conv2d", &zerograd::conv2d);
        m.def("maxPool2d", &zerograd::maxPool2d);
        m.def("flatten", &zerograd::flatten);

        m.def("mse_loss", &zerograd::mse_loss);
        m.def("ce_loss", &zerograd::ce_loss,
            py::arg("logits"), py::arg("target_classes"));

        // Layer class
        py::class_<zerograd::Layer, std::shared_ptr<zerograd::Layer>>(m, "Layer");

        // Linear class
        py::class_<zerograd::Linear, zerograd::Layer, std::shared_ptr<zerograd::Linear>>(m, "Linear")
            .def(py::init<std::size_t, std::size_t>(), py::arg("in_features"), py::arg("out_features"))
            .def("forward", &zerograd::Linear::forward)
            .def("parameters", &zerograd::Linear::parameters);

        // Hidden layer classes
        py::class_<zerograd::Conv2d, zerograd::Layer, std::shared_ptr<zerograd::Conv2d>>(m, "Conv2d")
            .def(py::init<std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t>(),
                py::arg("in_channels"), py::arg("out_channels"),
                py::arg("kernel_h"), py::arg("kernel_w"),
                py::arg("stride") = 1, py::arg("padding") = 0)
            .def("forward", &zerograd::Conv2d::forward)
            .def("parameters", &zerograd::Conv2d::parameters);

        py::class_<zerograd::MaxPool2d, zerograd::Layer, std::shared_ptr<zerograd::MaxPool2d>>(m, "MaxPool2d")
            .def(py::init<std::size_t, std::size_t, std::size_t, std::size_t>(),
                py::arg("kernel_h"), py::arg("kernel_w"),
                py::arg("stride") = 1, py::arg("padding") = 0)
            .def("forward", &zerograd::MaxPool2d::forward);

        py::class_<zerograd::Flatten, zerograd::Layer, std::shared_ptr<zerograd::Flatten>>(m, "Flatten")
            .def(py::init<>())
            .def("forward", &zerograd::Flatten::forward);

        // Activations
        py::class_<zerograd::ReLU, zerograd::Layer, std::shared_ptr<zerograd::ReLU>>(m, "ReLU")
            .def(py::init<>())
            .def("forward", &zerograd::ReLU::forward);

        py::class_<zerograd::Sigmoid, zerograd::Layer, std::shared_ptr<zerograd::Sigmoid>>(m, "Sigmoid")
            .def(py::init<>())
            .def("forward", &zerograd::Sigmoid::forward);

        py::class_<zerograd::Tanh, zerograd::Layer, std::shared_ptr<zerograd::Tanh>>(m, "Tanh")
            .def(py::init<>())
            .def("forward", &zerograd::Tanh::forward);

        // Sequential class
        py::class_<zerograd::Sequential, std::shared_ptr<zerograd::Sequential>>(m, "Sequential")
            .def(py::init<>())
            .def("forward", &zerograd::Sequential::forward)
            .def("parameters", &zerograd::Sequential::parameters)
            .def("add", &zerograd::Sequential::add);

        py::class_<zerograd::Optimizer, std::shared_ptr<zerograd::Optimizer>>(m, "Optimizer")
            .def(py::init<std::vector<std::shared_ptr<zerograd::Tensor>>, float>(), py::arg("parameters"), py::arg("lr"))
            .def("zero_grad", &zerograd::Optimizer::zero_grad)
            .def("step", &zerograd::Optimizer::step);

        // MNIST and DataLoader classes
        py::class_<zerograd::MNISTData>(m, "MNISTData")
            .def_readonly("images", &zerograd::MNISTData::images)
            .def_readonly("labels", &zerograd::MNISTData::labels)
            .def_readonly("num_samples", &zerograd::MNISTData::num_samples)
            .def_readonly("image_size", &zerograd::MNISTData::image_size);

        m.def("load_mnist", &zerograd::load_mnist,
            py::arg("images_path"), py::arg("labels_path"));

        py::class_<zerograd::DataLoader>(m, "DataLoader")
            .def(py::init<zerograd::MNISTData, std::size_t>(),
                py::arg("data"), py::arg("batch_size"))
            .def("shuffle", &zerograd::DataLoader::shuffle)
            .def("reset", &zerograd::DataLoader::reset)
            .def("has_next", &zerograd::DataLoader::has_next)
            .def("next_batch", &zerograd::DataLoader::next_batch);

        m.def("count_correct", &zerograd::metrics::count_correct,
            py::arg("logits"), py::arg("targets"));

        // Graph Analyzer
        py::class_<zerograd::GraphAnalyzer>(m, "GraphAnalyzer")
            .def(py::init<>())
            .def("dry_forward", &zerograd::GraphAnalyzer::dry_forward);
}