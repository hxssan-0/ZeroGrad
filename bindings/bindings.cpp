#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <zerograd/tensor.h>
#include <zerograd/layer.h>
#include <zerograd/linear.h>
#include <zerograd/sequential.h>
#include <zerograd/optimizer.h>
#include <zerograd/activations.h>
#include <zerograd/data/mnist.h>
#include <zerograd/metrics.h>

namespace py = pybind11;

PYBIND11_MODULE(_zerograd_backend, m)
{
    m.doc() = "ZeroGrad C++ Backend";

    // Tensor class
    py::class_<zerograd::Tensor, std::shared_ptr<zerograd::Tensor>>(m, "Tensor")
        .def(py::init<
                std::vector<float>,
                std::vector<std::size_t>,
                bool,
                std::vector<std::shared_ptr<zerograd::Tensor>>,
                std::string
                >(),
            py::arg("data"),
            py::arg("shape"),
            py::arg("requires_grad") = false,
            py::arg("children") = std::vector<std::shared_ptr<zerograd::Tensor>>{},
            py::arg("op") = ""
        )

        .def_readwrite("data", &zerograd::Tensor::data)
        .def_readwrite("shape", &zerograd::Tensor::shape)
        .def_readwrite("strides", &zerograd::Tensor::strides)
        .def_readwrite("grad", &zerograd::Tensor::grad)

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
}