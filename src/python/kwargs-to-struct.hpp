/**
 * @file
 * This file defines mappings from Python dicts (kwargs) to simple parameter
 * structs.
 */

#pragma once

#include <functional>
#include <map>
#include <variant>

#include <pybind11/detail/typeid.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

struct cast_error_with_types : py::cast_error {
    cast_error_with_types(const py::cast_error &e, std::string from,
                          std::string to)
        : py::cast_error(e), from(std::move(from)), to(std::move(to)) {}
    std::string from;
    std::string to;
};

template <class T, class A>
auto attr_setter(A T::*attr) {
    return [attr](T &t, const py::handle &h) {
        try {
            t.*attr = h.cast<A>();
        } catch (const py::cast_error &e) {
            throw cast_error_with_types(e, py::str(py::type::handle_of(h)),
                                        py::type_id<A>());
        }
    };
}
template <class T, class A>
auto attr_getter(A T::*attr) {
    return [attr](const T &t) { return py::cast(t.*attr); };
}

template <class T>
class attr_setter_fun_t {
  public:
    template <class A>
    attr_setter_fun_t(A T::*attr)
        : set(attr_setter(attr)), get(attr_getter(attr)) {}

    std::function<void(T &, const py::handle &)> set;
    std::function<py::object(const T &)> get;
};

template <class T>
using kwargs_to_struct_table_t = std::map<std::string, attr_setter_fun_t<T>>;

template <class T>
kwargs_to_struct_table_t<T> kwargs_to_struct_table;

template <class T>
void kwargs_to_struct_helper(T &t, const py::kwargs &kwargs) {
    const auto &m = kwargs_to_struct_table<T>;
    for (auto &&[key, val] : kwargs) {
        auto skey = key.template cast<std::string>();
        auto it   = m.find(skey);
        if (it == m.end())
            throw py::key_error("Unknown parameter " + skey);
        try {
            it->second.set(t, val);
        } catch (const cast_error_with_types &e) {
            throw std::runtime_error("Error converting parameter '" + skey +
                                     "' from " + e.from + " to '" + e.to +
                                     "': " + e.what());
        } catch (const std::runtime_error &e) {
            throw std::runtime_error("Error setting parameter '" + skey +
                                     "': " + e.what());
        }
    }
}

template <class T>
py::dict struct_to_dict_helper(const T &t) {
    const auto &m = kwargs_to_struct_table<T>;
    py::dict d;
    for (auto &&[key, val] : m) {
        py::object o = val.get(t);
        if (py::hasattr(o, "to_dict"))
            o = o.attr("to_dict")();
        d[key.c_str()] = std::move(o);
    }
    return d;
}

template <class T>
T kwargs_to_struct(const py::kwargs &kwargs) {
    T t{};
    kwargs_to_struct_helper(t, kwargs);
    return t;
}

template <class T>
py::dict struct_to_dict(const T &t) {
    return struct_to_dict_helper<T>(t);
}

template <class T>
T var_kwargs_to_struct(const std::variant<T, py::dict> &p) {
    return std::holds_alternative<T>(p)
               ? std::get<T>(p)
               : kwargs_to_struct<T>(std::get<py::dict>(p));
}

#include <quala/lbfgs.hpp>

template <>
inline const kwargs_to_struct_table_t<quala::LBFGSParams>
    kwargs_to_struct_table<quala::LBFGSParams>{
        {"memory", &quala::LBFGSParams::memory},
        {"min_div_fac", &quala::LBFGSParams::min_div_fac},
        {"min_abs_s", &quala::LBFGSParams::min_abs_s},
        {"force_pos_def", &quala::LBFGSParams::force_pos_def},
        {"cbfgs", &quala::LBFGSParams::cbfgs},
    };

template <>
inline const kwargs_to_struct_table_t<decltype(quala::LBFGSParams::cbfgs)>
    kwargs_to_struct_table<decltype(quala::LBFGSParams::cbfgs)>{
        {"α", &decltype(quala::LBFGSParams::cbfgs)::α},
        {"ϵ", &decltype(quala::LBFGSParams::cbfgs)::ϵ},
    };

#include <quala/anderson-acceleration.hpp>

template <>
inline const kwargs_to_struct_table_t<quala::AndersonAccelParams>
    kwargs_to_struct_table<quala::AndersonAccelParams>{
        {"memory", &quala::AndersonAccelParams::memory},
    };

#include <quala/broyden-good.hpp>

template <>
inline const kwargs_to_struct_table_t<quala::BroydenGoodParams>
    kwargs_to_struct_table<quala::BroydenGoodParams>{
        {"memory", &quala::BroydenGoodParams::memory},
        {"min_div_abs", &quala::BroydenGoodParams::min_div_abs},
        {"force_pos_def", &quala::BroydenGoodParams::force_pos_def},
        {"restarted", &quala::BroydenGoodParams::restarted},
    };
