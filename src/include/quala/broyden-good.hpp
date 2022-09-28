#pragma once

#include <quala/util/vec.hpp>

namespace quala {

/// Layout:
/// ~~~
///       ┌───── 2 m + 1 ─────┐
///     ┌ ┌───┬───┬───┬───┬───┐
///     │ │   │   │   │   │   │
///   n │ │ s │ s̃ │ s │ s̃ │ w │
///     │ │   │   │   │   │   │
///     └ └───┴───┴───┴───┴───┘
/// ~~~
struct BroydenStorage {
    /// Re-allocate storage for a problem with a different size.
    void resize(length_t n, length_t history);

    /// Get the size of the s and s̃ vectors in the buffer.
    length_t n() const { return sto.rows(); }
    /// Get the number of previous vectors s and s̃ stored in the buffer.
    length_t history() const { return (sto.cols() - 1) / 2; }

    auto s(index_t i) { return sto.col(2 * i); }
    auto s(index_t i) const { return sto.col(2 * i); }
    auto s̃(index_t i) { return sto.col(2 * i + 1); }
    auto s̃(index_t i) const { return sto.col(2 * i + 1); }
    auto work() { return sto.col(2 * history()); }
    auto work() const { return sto.col(2 * history()); }

    using storage_t =
        Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    storage_t sto;
};

inline void BroydenStorage::resize(length_t n, length_t history) {
    sto.resize(n, history * 2 + 1);
}

/// Parameters for the @ref BroydenGood class.
struct BroydenGoodParams {
    /// Length of the history to keep.
    length_t memory = 10;
    /// Reject update if @f$ s^\top Hy \le \text{min_div_fac} @f$.
    real_t min_div_abs = 1e-32;
    /// If set to true, the inverse Jacobian estimate should remain definite.
    bool force_pos_def = false;
    /// If set to true, the buffer is cleared after @p memory iterations. If
    /// set to false, a circular buffer is used, replacing the oldest pair of
    /// vectors, as a limited-memory type algorithm. However, since each vector
    /// s̃ depends on the previous vectors, this algorithm is not theoretically
    /// correct, because the s̃ vectors are not updated when the old history is
    /// overwritten.
    bool restarted = true;
    /// Powell's trick, damping, prevents nonsingularity.
    real_t powell_damping_factor = 0;
    /// Minimum automatic step size. If @f$ \frac{s^\top y}{y^\top y} @f$ is
    /// smaller than this setting, use this as the step size instead.
    real_t min_stepsize = 1e-10;
};

/**
 * Broyden's “Good” method for solving systems of nonlinear equations
 * @f$ F(x) = 0 @f$.
 *
 * @f[ \begin{aligned}
 *     B_{k+1} &= B_k + \frac{\left(y_k - B_k s_k\right) s_k^\top}
 *                           {s_k^\top s_k} \\
 *     H_{k+1} &= H_k + \frac{\left(s_k - H_k y_k\right)
 *                            \left(s_k^\top H_k\right)}
 *                           {s_k^\top H_k y_k} \\
 *     s_k &\triangleq x_{k+1} - x_k \\
 *     y_k &\triangleq F(x_{k+1}) - F(x_k) \\
 * \end{aligned} @f]
 * Where @f$ B_k @f$ approximates the Jacobian of @f$ F(x_k) @f$ and 
 * @f$ H_k \triangleq B_k^{-1} @f$.
 *
 * @todo    Damping.
 *
 * @ingroup accelerators-grp
 */
class BroydenGood {
  public:
    using Params = BroydenGoodParams;

    BroydenGood(Params params) : params(params) {}
    BroydenGood(Params params, length_t n) : params(params) { resize(n); }

    /// Update the inverse Jacobian approximation using the new vectors
    /// sₖ = xₖ₊₁ - xₖ and yₖ = pₖ₊₁ - pₖ.
    template <class VecS, class VecY>
    bool update_sy(const anymat<VecS> &s, const anymat<VecY> &y,
                   bool forced = false);

    /// Update the inverse Jacobian approximation using the new vectors xₖ₊₁
    /// and pₖ₊₁.
    bool update(crvec xₖ, crvec xₙₑₓₜ, crvec pₖ, crvec pₙₑₓₜ,
                bool forced = false);

    /// Apply the inverse Jacobian approximation to the given vector q, i.e.
    /// @f$ q \leftarrow H_k q @f$.
    /// Initial inverse Hessian approximation is set to @f$ H_0 = \gamma I @f$.
    /// The result is scaled by a factor @p γ. If @p γ is negative, the result
    /// is scaled by @f$ \frac{s^\top y}{y^\top y} @f$.
    bool apply(rvec q, real_t γ);

    /// Throw away the approximation and all previous vectors s and y.
    void reset();
    /// Re-allocate storage for a problem with a different size. Causes
    /// a @ref reset.
    void resize(length_t n);

    /// Get the parameters.
    const Params &get_params() const { return params; }

    /// Get the size of the s and y vectors in the buffer.
    length_t n() const { return sto.n(); }
    /// Get the number of previous vectors s and y stored in the buffer.
    length_t history() const { return sto.history(); }
    /// Get the next index in the circular buffer of previous s and y vectors.
    index_t succ(index_t i) const { return i + 1 < history() ? i + 1 : 0; }
    /// Get the previous index in the circular buffer of s and y vectors.
    index_t pred(index_t i) const { return i > 0 ? i - 1 : history() - 1; }
    /// Get the number of previous s and y vectors currently stored in the
    /// buffer.
    length_t current_history() const { return full ? history() : idx; }

    auto s(index_t i) { return sto.s(i); }
    auto s(index_t i) const { return sto.s(i); }
    auto s̃(index_t i) { return sto.s̃(i); }
    auto s̃(index_t i) const { return sto.s̃(i); }

    /// Iterate over the indices in the history buffer, oldest first.
    template <class F>
    void foreach_fwd(const F &fun) const {
        if (full)
            for (index_t i = idx; i < history(); ++i)
                fun(i);
        if (idx)
            for (index_t i = 0; i < idx; ++i)
                fun(i);
    }

    /// Iterate over the indices in the history buffer, newest first.
    template <class F>
    void foreach_rev(const F &fun) const {
        if (idx)
            for (index_t i = idx; i-- > 0;)
                fun(i);
        if (full)
            for (index_t i = history(); i-- > idx;)
                fun(i);
    }

  private:
    BroydenStorage sto;
    index_t idx = 0;
    bool full   = false;
    Params params;
    real_t latest_γ = NaN;
};

inline void BroydenGood::reset() {
    idx  = 0;
    full = false;
}

inline void BroydenGood::resize(length_t n) {
    if (params.memory < 1)
        throw std::invalid_argument("BroydenGood::Params::memory must be >= 1");
    sto.resize(n, params.memory);
    reset();
}

template <class VecS, class VecY>
bool BroydenGood::update_sy(const anymat<VecS> &sₖ, const anymat<VecY> &yₖ,
                            bool forced) {
    // Restart if the buffer is full
    if (full && params.restarted) {
        full = false;
        assert(idx == 0);
    }

    auto &&r = sto.work();
    // Compute r = r₍ₘ₋₁₎ = Hₖ yₖ
    r = yₖ; // r₍₋₁₎ = yₖ
    foreach_fwd([&](index_t i) {
        r += s̃(i) * r.dot(s(i)); // r₍ᵢ₎ = r₍ᵢ₋₁₎ + s̃₍ᵢ₎〈r₍ᵢ₋₁₎, s₍ᵢ₎〉
    });
    const real_t sᵀHy   = sₖ.dot(r);
    const real_t a_sᵀHy = params.force_pos_def ? sᵀHy : std::abs(sᵀHy);

    // Check if update is accepted
    if (!forced && a_sᵀHy < params.min_div_abs)
        return false;

    // Compute damping
    real_t damp = 1 / sᵀHy;
    if (real_t θ̅ = params.powell_damping_factor) {
        real_t γ     = sᵀHy / sₖ.squaredNorm();
        real_t sgn_γ = γ >= 0 ? 1 : -1;
        real_t γθ̅    = params.force_pos_def ? γ * θ̅ : sgn_γ * θ̅;
        real_t a_γ   = params.force_pos_def ? γ : std::abs(γ);
        real_t θ     = a_γ >= θ̅ ? 1 // no damping
                                : (1 - γθ̅) / (1 - γ);
        damp         = θ / (sₖ.squaredNorm() * (1 - θ + θ * γ));
    }

    // Store the new vectors
    sto.s(idx) = sₖ;
    sto.s̃(idx) = damp * (sₖ - r);
    latest_γ = sₖ.dot(yₖ) / yₖ.squaredNorm();
    if (std::abs(latest_γ) < params.min_stepsize)
        latest_γ = std::copysign(params.min_stepsize, latest_γ);

    // Increment the index in the circular buffer
    idx = succ(idx);
    full |= idx == 0;

    return true;
}

inline bool BroydenGood::apply(rvec q, real_t γ) {
    // Only apply if we have previous vectors s and y
    if (idx == 0 && not full)
        return false;

    if (γ < 0)
        γ = latest_γ;
    if (γ != 1)
        q *= γ;

    // Compute q = q₍ₘ₋₁₎ = Hₖ q
    // q₍₋₁₎ = q
    foreach_fwd([&](index_t i) {
        q += s̃(i) * q.dot(s(i)); // q₍ᵢ₎ = q₍ᵢ₋₁₎ + s̃₍ᵢ₎〈q₍ᵢ₋₁₎, s₍ᵢ₎〉
    });

    return true;
}

inline bool BroydenGood::update(crvec xₖ, crvec xₙₑₓₜ, crvec pₖ, crvec pₙₑₓₜ,
                                bool forced) {
    const auto s = xₙₑₓₜ - xₖ;
    const auto y = pₙₑₓₜ - pₖ;
    return update_sy(s, y, forced);
}

} // namespace quala