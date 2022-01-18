#pragma once

#include <Eigen/Core>

namespace quala {

/// Default floating point type
using real_t = double; // TODO: make template?
/// Default type for floating point vectors.
using realvec = Eigen::Matrix<real_t, Eigen::Dynamic, 1>;
/// Default type for floating point matrices.
using realmat = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic>;
/// Default type for vectors.
using vec = realvec;
/// Default type for mutable references to vectors.
using rvec = Eigen::Ref<vec>;
/// Default type for immutable references to vectors.
using crvec = Eigen::Ref<const vec>;
/// Default type for matrices.
using mat = realmat;
/// Default type for mutable references to matrices.
using rmat = Eigen::Ref<mat>;
/// Default type for immutable references to matrices.
using crmat = Eigen::Ref<const mat>;

/// Default type for vector indices.
using index_t = Eigen::Index;
/// Default type for vector sizes.
using length_t = index_t;

/// Type for a vector of indices.
using idvec = Eigen::Matrix<index_t, Eigen::Dynamic, 1>;
/// Mutable reference to vector indices.
using ridvec = Eigen::Ref<idvec>;
/// Immutable reference to vector indices.
using cridvec = Eigen::Ref<const idvec>;

/// Generic type for vector and matrix arguments.
template <class Derived>
using anymat = Eigen::MatrixBase<Derived>;

/// @f$ \infty @f$
constexpr real_t inf = std::numeric_limits<real_t>::infinity();
/// Not a number.
constexpr real_t NaN = std::numeric_limits<real_t>::quiet_NaN();

namespace vec_util {

/// Get the Σ norm squared of a given vector, with Σ a diagonal matrix.
/// @returns @f$ \langle v, \Sigma v \rangle @f$
template <class V, class M>
auto norm_squared_weighted(V &&v, M &&Σ) {
    return v.dot(Σ.asDiagonal() * v);
}

/// Get the maximum or infinity-norm of the given vector.
/// @returns @f$ \left\|v\right\|_\infty @f$
template <class Vec>
real_t norm_inf(const Vec &v) {
    return v.template lpNorm<Eigen::Infinity>();
}

/// Get the 1-norm of the given vector.
/// @returns @f$ \left\|v\right\|_1 @f$
template <class Vec>
real_t norm_1(const Vec &v) {
    return v.template lpNorm<1>();
}

} // namespace vec_util

} // namespace quala