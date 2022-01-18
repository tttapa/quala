#pragma once

#include <quala/decl/lbfgs-fwd.hpp>
#include <quala/util/vec.hpp>

namespace quala {

/// Parameters for the @ref LBFGS class.
struct LBFGSParams {
    /// Length of the history to keep.
    length_t memory = 10;
    /// Reject update if @f$ y^\top s \le \text{min_div_fac} \cdot s^\top s @f$.
    real_t min_div_fac = 1e-10;
    /// Reject update if @f$ s^\top s \le \text{min_abs_s} @f$.
    real_t min_abs_s = 1e-32;
    /// Cautious BFGS update.
    /// @see @ref cbfgs
    struct CBFGSParams {
        real_t α = 1;
        real_t ϵ = 0; ///< Set to zero to disable CBFGS check.
        explicit operator bool() const { return ϵ > 0; }
    };
    /// Parameters in the cautious BFGS update condition
    /// @f[ \frac{y^\top s}{s^\top s} \ge \epsilon \| g \|^\alpha @f]
    /// @see https://epubs.siam.org/doi/10.1137/S1052623499354242
    CBFGSParams cbfgs;
    /// If set to true, the inverse Hessian estimate should remain definite,
    /// i.e. a check is performed that rejects the update if
    /// @f$ y^\top s \le \text{min_div_fac} \cdot s^\top s @f$.
    /// If set to false, just try to prevent a singular Hessian by rejecting the
    /// update if
    /// @f$ \left| y^\top s \right| \le \text{min_div_fac} \cdot s^\top s @f$.
    bool force_pos_def = true;
};

/// Layout:
/// ~~~
///       ┌───── 2 m ─────┐
///     ┌ ┌───┬───┬───┬───┐
///     │ │   │   │   │   │
///     │ │ s │ y │ s │ y │
/// n+1 │ │   │   │   │   │
///     │ ├───┼───┼───┼───┤
///     │ │ ρ │ α │ ρ │ α │
///     └ └───┴───┴───┴───┘
/// ~~~
struct LBFGSStorage {
    /// Re-allocate storage for a problem with a different size.
    void resize(length_t n, length_t history);

    /// Get the size of the s and y vectors in the buffer.
    length_t n() const { return sto.rows() - 1; }
    /// Get the number of previous vectors s and y stored in the buffer.
    length_t history() const { return sto.cols() / 2; }

    auto s(index_t i) { return sto.col(2 * i).topRows(n()); }
    auto s(index_t i) const { return sto.col(2 * i).topRows(n()); }
    auto y(index_t i) { return sto.col(2 * i + 1).topRows(n()); }
    auto y(index_t i) const { return sto.col(2 * i + 1).topRows(n()); }
    real_t &ρ(index_t i) { return sto.coeffRef(n(), 2 * i); }
    const real_t &ρ(index_t i) const { return sto.coeff(n(), 2 * i); }
    real_t &α(index_t i) { return sto.coeffRef(n(), 2 * i + 1); }
    const real_t &α(index_t i) const { return sto.coeff(n(), 2 * i + 1); }

    using storage_t =
        Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    storage_t sto;
};

/// Limited memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) algorithm
/// @ingroup accelerators-grp
class LBFGS {
  public:
    using Params = LBFGSParams;

    /// The sign of the vectors @f$ p @f$ passed to the @ref LBFGS::update
    /// method.
    enum class Sign {
        Positive, ///< @f$ p \sim \nabla \psi(x) @f$
        Negative, ///< @f$ p \sim -\nabla \psi(x) @f$
    };

    LBFGS(Params params) : params(params) {}
    LBFGS(Params params, length_t n) : params(params) { resize(n); }

    /// Check if the new vectors s and y allow for a valid BFGS update that
    /// preserves the positive definiteness of the Hessian approximation.
    static bool update_valid(const LBFGSParams &params, real_t yᵀs, real_t sᵀs,
                             real_t pᵀp);

    /// Update the inverse Hessian approximation using the new vectors
    /// sₖ = xₖ₊₁ - xₖ and yₖ = pₖ₊₁ - pₖ.
    template <class VecS, class VecY>
    bool update_sy(const anymat<VecS> &s, const anymat<VecY> &y,
                   real_t pₖ₊₁ᵀpₖ₊₁, bool forced = false);

    /// Update the inverse Hessian approximation using the new vectors xₖ₊₁
    /// and pₖ₊₁.
    bool update(crvec xₖ, crvec xₖ₊₁, crvec pₖ, crvec pₖ₊₁,
                Sign sign = Sign::Positive, bool forced = false);

    /// Apply the inverse Hessian approximation to the given vector q.
    /// Initial inverse Hessian approximation is set to @f$ H_0 = \gamma I @f$.
    /// If @p γ is negative, @f$ H_0 = \frac{s^\top y}{y^\top y} I @f$.
    bool apply(rvec q, real_t γ);

    /// Apply the inverse Hessian approximation to the given vector q, applying
    /// only the columns and rows of the Hessian in the index set J.
    template <class IndexVec>
    bool apply(rvec q, real_t γ, const IndexVec &J);

    /// Throw away the approximation and all previous vectors s and y.
    void reset();
    /// Re-allocate storage for a problem with a different size. Causes
    /// a @ref reset.
    void resize(length_t n);

    /// Scale the stored y vectors by the given factor.
    void scale_y(real_t factor);

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
    auto y(index_t i) { return sto.y(i); }
    auto y(index_t i) const { return sto.y(i); }
    real_t &ρ(index_t i) { return sto.ρ(i); }
    const real_t &ρ(index_t i) const { return sto.ρ(i); }
    real_t &α(index_t i) { return sto.α(i); }
    const real_t &α(index_t i) const { return sto.α(i); }

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
    LBFGSStorage sto;
    index_t idx = 0;
    bool full   = false;
    Params params;
};

} // namespace quala