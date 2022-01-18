/**
 * @file
 * This file defines all Python bindings.
 */

#include <pybind11/attr.h>
#include <pybind11/cast.h>
#include <pybind11/chrono.h>
#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "kwargs-to-struct.hpp"

namespace py = pybind11;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

PYBIND11_MODULE(QUALA_MODULE_NAME, m) {
    using py::operator""_a;

    py::options options;
    options.enable_function_signatures();
    options.enable_user_defined_docstrings();

    m.doc() = "Quala Quasi-Newton algorithms";

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    py::class_<quala::LBFGSParams::CBFGSParams>(
        m, "LBFGSParamsCBFGS",
        "C++ documentation: :cpp:member:`quala::LBFGSParams::CBFGSParams `")
        .def(py::init())
        .def(py::init(&kwargs_to_struct<quala::LBFGSParams::CBFGSParams>))
        .def("to_dict", &struct_to_dict<quala::LBFGSParams::CBFGSParams>)
        .def_readwrite("α", &quala::LBFGSParams::CBFGSParams::α)
        .def_readwrite("ϵ", &quala::LBFGSParams::CBFGSParams::ϵ);

    py::class_<quala::LBFGSParams>(
        m, "LBFGSParams", "C++ documentation: :cpp:class:`quala::LBFGSParams`")
        .def(py::init())
        .def(py::init(&kwargs_to_struct<quala::LBFGSParams>))
        .def("to_dict", &struct_to_dict<quala::LBFGSParams>)
        .def_readwrite("memory", &quala::LBFGSParams::memory)
        .def_readwrite("cbfgs", &quala::LBFGSParams::cbfgs);

    auto lbfgs = py::class_<quala::LBFGS>(
        m, "LBFGS", "C++ documentation: :cpp:class:`quala::LBFGS`");
    auto lbfgssign = py::enum_<quala::LBFGS::Sign>(
        lbfgs, "Sign", "C++ documentation :cpp:enum:`quala::LBFGS::Sign`");
    lbfgssign //
        .value("Positive", quala::LBFGS::Sign::Positive)
        .value("Negative", quala::LBFGS::Sign::Negative)
        .export_values();
    lbfgs //
        .def(py::init<quala::LBFGS::Params>(), "params"_a)
        .def(py::init([](py::dict params) -> quala::LBFGS {
                 return {kwargs_to_struct<quala::LBFGS::Params>(params)};
             }),
             "params"_a)
        .def(py::init<quala::LBFGS::Params, quala::length_t>(), "params"_a,
             "n"_a)
        .def(py::init([](py::dict params, quala::length_t n) -> quala::LBFGS {
                 return {kwargs_to_struct<quala::LBFGS::Params>(params), n};
             }),
             "params"_a, "n"_a)
        .def_static("update_valid", quala::LBFGS::update_valid, "params"_a,
                    "yᵀs"_a, "sᵀs"_a, "pᵀp"_a)
        .def("update", &quala::LBFGS::update, "xk"_a, "xkp1"_a, "pk"_a,
             "pkp1"_a, "sign"_a, "forced"_a = false)
        .def(
            "apply",
            [](quala::LBFGS &lbfgs, quala::rvec q, quala::real_t γ) {
                return lbfgs.apply(q, γ);
            },
            "q"_a, "γ"_a)
        .def(
            "apply",
            [](quala::LBFGS &lbfgs, quala::rvec q, quala::real_t γ,
               const std::vector<quala::vec::Index> &J) {
                return lbfgs.apply(q, γ, J);
            },
            "q"_a, "γ"_a, "J"_a)
        .def("reset", &quala::LBFGS::reset)
        .def("resize", &quala::LBFGS::resize, "n"_a)
        .def("scale_y", &quala::LBFGS::scale_y, "factor"_a)
        .def_property_readonly("n", &quala::LBFGS::n)
        .def("s",
             [](quala::LBFGS &lbfgs, quala::index_t i) -> quala::rvec {
                 return lbfgs.s(i);
             })
        .def("y",
             [](quala::LBFGS &lbfgs, quala::index_t i) -> quala::rvec {
                 return lbfgs.y(i);
             })
        .def("ρ",
             [](quala::LBFGS &lbfgs, quala::index_t i) -> quala::real_t & {
                 return lbfgs.ρ(i);
             })
        .def("α",
             [](quala::LBFGS &lbfgs, quala::index_t i) -> quala::real_t & {
                 return lbfgs.α(i);
             })
        .def_property_readonly("params", &quala::LBFGS::get_params);

    py::class_<quala::AndersonAccelParams>(
        m, "AndersonAccelParams",
        "C++ documentation: :cpp:class:`quala::AndersonAccelParams`")
        .def(py::init())
        .def(py::init(&kwargs_to_struct<quala::AndersonAccelParams>))
        .def("to_dict", &struct_to_dict<quala::AndersonAccelParams>)
        .def_readwrite("memory", &quala::AndersonAccelParams::memory);

    py::class_<quala::AndersonAccel>(
        m, "AndersonAccel",
        "C++ documentation: :cpp:class:`quala::AndersonAccel`")
        .def(py::init<quala::AndersonAccel::Params>(), "params"_a)
        .def(py::init([](py::dict params) -> quala::AndersonAccel {
                 return {
                     kwargs_to_struct<quala::AndersonAccel::Params>(params)};
             }),
             "params"_a)
        .def(py::init<quala::AndersonAccel::Params, quala::length_t>(),
             "params"_a, "n"_a)
        .def(py::init([](py::dict params,
                         quala::length_t n) -> quala::AndersonAccel {
                 return {kwargs_to_struct<quala::AndersonAccel::Params>(params),
                         n};
             }),
             "params"_a, "n"_a)
        .def("resize", &quala::AndersonAccel::resize, "n"_a)
        .def(
            "initialize",
            [](quala::AndersonAccel &self, quala::crvec g_0, quala::vec r_0) {
                if (g_0.size() != quala::vec::Index(self.n()))
                    throw std::invalid_argument("g_0 dimension mismatch");
                if (r_0.size() != quala::vec::Index(self.n()))
                    throw std::invalid_argument("r_0 dimension mismatch");
                self.initialize(g_0, std::move(r_0));
            },
            "g_0"_a, "r_0"_a)
        .def(
            "compute",
            [](quala::AndersonAccel &self, quala::crvec g_k, quala::vec r_k,
               quala::rvec x_k_aa) {
                if (g_k.size() != quala::vec::Index(self.n()))
                    throw std::invalid_argument("g_k dimension mismatch");
                if (r_k.size() != quala::vec::Index(self.n()))
                    throw std::invalid_argument("r_k dimension mismatch");
                if (x_k_aa.size() != quala::vec::Index(self.n()))
                    throw std::invalid_argument("x_k_aa dimension mismatch");
                self.compute(g_k, std::move(r_k), x_k_aa);
            },
            "g_k"_a, "r_k"_a, "x_k_aa"_a)
        .def(
            "compute",
            [](quala::AndersonAccel &self, quala::crvec g_k, quala::vec r_k) {
                if (g_k.size() != quala::vec::Index(self.n()))
                    throw std::invalid_argument("g_k dimension mismatch");
                if (r_k.size() != quala::vec::Index(self.n()))
                    throw std::invalid_argument("r_k dimension mismatch");
                quala::vec x_k_aa(self.n());
                self.compute(g_k, std::move(r_k), x_k_aa);
                return x_k_aa;
            },
            "g_k"_a, "r_k"_a)
        .def("reset", &quala::AndersonAccel::reset)
        .def_property_readonly("params", &quala::AndersonAccel::get_params);
}