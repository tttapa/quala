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

PYBIND11_MODULE(QUALA_MODULE_NAME, m) {
    using py::operator""_a;
    using quala::crmat;
    using quala::crvec;
    using quala::index_t;
    using quala::length_t;
    using quala::mat;
    using quala::real_t;
    using quala::rmat;
    using quala::rvec;
    using quala::vec;

    py::options options;
    options.enable_function_signatures();
    options.enable_user_defined_docstrings();

    m.doc() = "Quala Quasi-Newton algorithms";

#ifdef QUALA_VERSION_INFO
    m.attr("__version__") = QUALA_VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
    using quala::LimitedMemoryQR;
    py::class_<LimitedMemoryQR>(m, "LimitedMemoryQR")
        .def(py::init<>())
        .def(py::init<length_t, length_t>(), "n"_a, "m"_a)
        .def_property_readonly("n", &LimitedMemoryQR::n)
        .def_property_readonly("m", &LimitedMemoryQR::m)
        .def_property_readonly("size", &LimitedMemoryQR::size)
        .def_property_readonly("history", &LimitedMemoryQR::history)
        .def("add_column", &LimitedMemoryQR::add_column<crvec>, "v"_a)
        .def("remove_column", &LimitedMemoryQR::remove_column)
        .def(
            "solve",
            [](const LimitedMemoryQR &qr, crmat b, rmat x, real_t tol) { qr.solve(b, x, tol); },
            "b"_a, "x"_a, "tol"_a = 0)
        .def(
            "solve",
            [](const LimitedMemoryQR &qr, crmat b, real_t tol) -> mat {
                mat x(qr.num_columns(), b.cols());
                qr.solve(b, x, tol);
                return x;
            },
            "b"_a, "tol"_a = 0)
        .def_property_readonly("R", &LimitedMemoryQR::get_R)
        .def_property_readonly("Q", &LimitedMemoryQR::get_Q)
        .def("reset", &LimitedMemoryQR::reset)
        .def("resize", &LimitedMemoryQR::resize)
        .def_property_readonly("num_columns", &LimitedMemoryQR::num_columns)
        .def_property_readonly("min_eig", &LimitedMemoryQR::get_min_eig)
        .def_property_readonly("max_eig", &LimitedMemoryQR::get_max_eig)
        .def_property_readonly("current_history", &LimitedMemoryQR::current_history)
        .def_property_readonly("reorth_count", &LimitedMemoryQR::get_reorth_count)
        .def("clear_reorth_count", &LimitedMemoryQR::clear_reorth_count);

    py::class_<quala::LBFGSParams::CBFGSParams>(
        m, "LBFGSParamsCBFGS", "C++ documentation: :cpp:member:`quala::LBFGSParams::CBFGSParams `")
        .def(py::init())
        .def(py::init(&kwargs_to_struct<quala::LBFGSParams::CBFGSParams>))
        .def("to_dict", &struct_to_dict<quala::LBFGSParams::CBFGSParams>)
        .def_readwrite("α", &quala::LBFGSParams::CBFGSParams::α)
        .def_readwrite("ϵ", &quala::LBFGSParams::CBFGSParams::ϵ)
        .def("__bool__", &quala::LBFGSParams::CBFGSParams::operator bool);

    py::class_<quala::LBFGSParams>(m, "LBFGSParams",
                                   "C++ documentation: :cpp:class:`quala::LBFGSParams`")
        .def(py::init())
        .def(py::init(&kwargs_to_struct<quala::LBFGSParams>))
        .def("to_dict", &struct_to_dict<quala::LBFGSParams>)
        .def_readwrite("memory", &quala::LBFGSParams::memory)
        .def_readwrite("min_div_fac", &quala::LBFGSParams::min_div_fac)
        .def_readwrite("min_abs_s", &quala::LBFGSParams::min_abs_s)
        .def_readwrite("force_pos_def", &quala::LBFGSParams::force_pos_def)
        .def_readwrite("cbfgs", &quala::LBFGSParams::cbfgs);

    auto lbfgs =
        py::class_<quala::LBFGS>(m, "LBFGS", "C++ documentation: :cpp:class:`quala::LBFGS`");
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
        .def(py::init<quala::LBFGS::Params, length_t>(), "params"_a, "n"_a)
        .def(py::init([](py::dict params, length_t n) -> quala::LBFGS {
                 return {kwargs_to_struct<quala::LBFGS::Params>(params), n};
             }),
             "params"_a, "n"_a)
        .def_static("update_valid", quala::LBFGS::update_valid, "params"_a, "yᵀs"_a, "sᵀs"_a,
                    "pᵀp"_a)
        .def(
            "update",
            [](quala::LBFGS &self, crvec xk, crvec xkp1, crvec pk, crvec pkp1,
               quala::LBFGS::Sign sign, bool forced) {
                if (xk.size() != self.n())
                    throw std::invalid_argument("xk dimension mismatch");
                if (xkp1.size() != self.n())
                    throw std::invalid_argument("xkp1 dimension mismatch");
                if (pk.size() != self.n())
                    throw std::invalid_argument("pk dimension mismatch");
                if (pkp1.size() != self.n())
                    throw std::invalid_argument("pkp1 dimension mismatch");
                return self.update(xk, xkp1, pk, pkp1, sign, forced);
            },
            "xk"_a, "xkp1"_a, "pk"_a, "pkp1"_a, "sign"_a = quala::LBFGS::Sign::Positive,
            "forced"_a = false)
        .def(
            "update_sy",
            [](quala::LBFGS &self, crvec sk, crvec yk, real_t pkp1Tpkp1, bool forced) {
                if (sk.size() != self.n())
                    throw std::invalid_argument("sk dimension mismatch");
                if (yk.size() != self.n())
                    throw std::invalid_argument("yk dimension mismatch");
                return self.update_sy(sk, yk, pkp1Tpkp1, forced);
            },
            "sk"_a, "yk"_a, "pkp1Tpkp1"_a, "forced"_a = false)
        .def(
            "apply",
            [](quala::LBFGS &self, rvec q, real_t γ) {
                if (q.size() != self.n())
                    throw std::invalid_argument("q dimension mismatch");
                return self.apply(q, γ);
            },
            "q"_a, "γ"_a)
        .def(
            "apply",
            [](quala::LBFGS &self, rvec q, real_t γ, const std::vector<vec::Index> &J) {
                return self.apply(q, γ, J);
            },
            "q"_a, "γ"_a, "J"_a)
        .def("reset", &quala::LBFGS::reset)
        .def("current_history", &quala::LBFGS::current_history)
        .def("resize", &quala::LBFGS::resize, "n"_a)
        .def("scale_y", &quala::LBFGS::scale_y, "factor"_a)
        .def_property_readonly("n", &quala::LBFGS::n)
        .def("s", [](quala::LBFGS &self, index_t i) -> rvec { return self.s(i); })
        .def("y", [](quala::LBFGS &self, index_t i) -> rvec { return self.y(i); })
        .def("ρ", [](quala::LBFGS &self, index_t i) -> real_t & { return self.ρ(i); })
        .def("α", [](quala::LBFGS &self, index_t i) -> real_t & { return self.α(i); })
        .def_property_readonly("params", &quala::LBFGS::get_params);

    py::class_<quala::AndersonAccelParams>(
        m, "AndersonAccelParams", "C++ documentation: :cpp:class:`quala::AndersonAccelParams`")
        .def(py::init())
        .def(py::init(&kwargs_to_struct<quala::AndersonAccelParams>))
        .def("to_dict", &struct_to_dict<quala::AndersonAccelParams>)
        .def_readwrite("memory", &quala::AndersonAccelParams::memory)
        .def_readwrite("min_div", &quala::AndersonAccelParams::min_div);

    py::class_<quala::AndersonAccel>(m, "AndersonAccel",
                                     "C++ documentation: :cpp:class:`quala::AndersonAccel`")
        .def(py::init<quala::AndersonAccel::Params>(), "params"_a)
        .def(py::init([](py::dict params) -> quala::AndersonAccel {
                 return {kwargs_to_struct<quala::AndersonAccel::Params>(params)};
             }),
             "params"_a)
        .def(py::init<quala::AndersonAccel::Params, length_t>(), "params"_a, "n"_a)
        .def(py::init([](py::dict params, length_t n) -> quala::AndersonAccel {
                 return {kwargs_to_struct<quala::AndersonAccel::Params>(params), n};
             }),
             "params"_a, "n"_a)
        .def("resize", &quala::AndersonAccel::resize, "n"_a)
        .def(
            "initialize",
            [](quala::AndersonAccel &self, crvec g_0, vec r_0) {
                if (g_0.size() != self.n())
                    throw std::invalid_argument("g_0 dimension mismatch");
                if (r_0.size() != self.n())
                    throw std::invalid_argument("r_0 dimension mismatch");
                self.initialize(g_0, std::move(r_0));
            },
            "g_0"_a, "r_0"_a)
        .def(
            "compute_inplace",
            [](quala::AndersonAccel &self, crvec g_k, vec r_k, rvec x_k_aa) {
                if (g_k.size() != self.n())
                    throw std::invalid_argument("g_k dimension mismatch");
                if (r_k.size() != self.n())
                    throw std::invalid_argument("r_k dimension mismatch");
                if (x_k_aa.size() != self.n())
                    throw std::invalid_argument("x_k_aa dimension mismatch");
                self.compute(g_k, std::move(r_k), x_k_aa);
            },
            "g_k"_a, "r_k"_a, "x_k_aa"_a)
        .def(
            "compute",
            [](quala::AndersonAccel &self, crvec g_k, vec r_k) {
                if (g_k.size() != self.n())
                    throw std::invalid_argument("g_k dimension mismatch");
                if (r_k.size() != self.n())
                    throw std::invalid_argument("r_k dimension mismatch");
                quala::vec x_k_aa(self.n());
                self.compute(g_k, std::move(r_k), x_k_aa);
                return x_k_aa;
            },
            "g_k"_a, "r_k"_a)
        .def("reset", &quala::AndersonAccel::reset)
        .def("current_history", &quala::AndersonAccel::current_history)
        .def_property_readonly("params", &quala::AndersonAccel::get_params);

    py::class_<quala::BroydenGoodParams>(m, "BroydenGoodParams",
                                         "C++ documentation: :cpp:class:`quala::BroydenGoodParams`")
        .def(py::init())
        .def(py::init(&kwargs_to_struct<quala::BroydenGoodParams>))
        .def("to_dict", &struct_to_dict<quala::BroydenGoodParams>)
        .def_readwrite("memory", &quala::BroydenGoodParams::memory)
        .def_readwrite("min_div_abs", &quala::BroydenGoodParams::min_div_abs)
        .def_readwrite("force_pos_def", &quala::BroydenGoodParams::force_pos_def)
        .def_readwrite("restarted", &quala::BroydenGoodParams::restarted)
        .def_readwrite("powell_damping_factor", &quala::BroydenGoodParams::powell_damping_factor)
        .def_readwrite("min_stepsize", &quala::BroydenGoodParams::min_stepsize);

    py::class_<quala::BroydenGood>(m, "BroydenGood",
                                   "C++ documentation: :cpp:class:`quala::BroydenGood`")
        .def(py::init<quala::BroydenGood::Params>(), "params"_a)
        .def(py::init([](py::dict params) -> quala::BroydenGood {
                 return {kwargs_to_struct<quala::BroydenGood::Params>(params)};
             }),
             "params"_a)
        .def(py::init<quala::BroydenGood::Params, length_t>(), "params"_a, "n"_a)
        .def(py::init([](py::dict params, length_t n) -> quala::BroydenGood {
                 return {kwargs_to_struct<quala::BroydenGood::Params>(params), n};
             }),
             "params"_a, "n"_a)
        .def("resize", &quala::BroydenGood::resize, "n"_a)
        .def(
            "update",
            [](quala::BroydenGood &self, crvec xk, crvec xkp1, crvec pk, crvec pkp1, bool forced) {
                if (xk.size() != self.n())
                    throw std::invalid_argument("xk dimension mismatch");
                if (xkp1.size() != self.n())
                    throw std::invalid_argument("xkp1 dimension mismatch");
                if (pk.size() != self.n())
                    throw std::invalid_argument("pk dimension mismatch");
                if (pkp1.size() != self.n())
                    throw std::invalid_argument("pkp1 dimension mismatch");
                return self.update(xk, xkp1, pk, pkp1, forced);
            },
            "xk"_a, "xkp1"_a, "pk"_a, "pkp1"_a, "forced"_a = false)
        .def(
            "update_sy",
            [](quala::BroydenGood &self, crvec sk, crvec yk, bool forced) {
                if (sk.size() != self.n())
                    throw std::invalid_argument("sk dimension mismatch");
                if (yk.size() != self.n())
                    throw std::invalid_argument("yk dimension mismatch");
                return self.update_sy(sk, yk, forced);
            },
            "sk"_a, "yk"_a, "forced"_a = false)
        .def(
            "apply",
            [](quala::BroydenGood &self, rvec q, real_t γ) {
                if (q.size() != self.n())
                    throw std::invalid_argument("q dimension mismatch");
                return self.apply(q, γ);
            },
            "q"_a, "γ"_a = -1)
        .def("reset", &quala::BroydenGood::reset)
        .def("current_history", &quala::BroydenGood::current_history)
        .def_property_readonly("params", &quala::BroydenGood::get_params);
}
