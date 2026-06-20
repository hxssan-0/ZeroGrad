#include <pybind11/pybind11.h>
#include <zerograd/scalar.h>

namespace py = pybind11;

PYBIND11_MODULE(_zerograd_backend, m) {
    m.doc() = "ZeroGrad C++ Backend";

    py::class_<zerograd::Scalar>(m, "Scalar")
        .def(py::init<float>())
        .def_readwrite("data", &zerograd::Scalar::data);
}