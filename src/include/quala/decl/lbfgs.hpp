#pragma once

#include <quala/decl/lbfgs-fwd.hpp>
#include <quala/util/vec.hpp>

namespace quala {

/// Parameters for the @ref LBFGS class.
struct LBFGSParams {
    /// Length of the history to keep.
    length_t memory = 10;
    struct CBFGSParams {
        real_t α = 1;
        real_t ϵ = 0;
    };
    /// Parameters in the cautious BFGS update condition
    /// @f[ \frac{y^\top s}{s^\top s} \ge \epsilon \| g \|^\alpha @f]
    /// @see https://epubs.siam.org/doi/10.1137/S1052623499354242
    CBFGSParams cbfgs;
};

/// Limited memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) algorithm
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
    static bool update_valid(LBFGSParams params, real_t yᵀs, real_t sᵀs,
                             real_t pᵀp);

    /// Update the inverse Hessian approximation using the new vectors xₖ₊₁
    /// and pₖ₊₁.
    bool update(crvec xₖ, crvec xₖ₊₁, crvec pₖ, crvec pₖ₊₁, Sign sign,
                bool forced = false);

    /// Apply the inverse Hessian approximation to the given vector q.
    /// Initial inverse Hessian approximation is set to @f$ H_0 = \gamma I @f$.
    /// If @p γ is negative, @f$ H_0 = \frac{s^\top y}{y^\top y} I @f$.
    template <class Vec>
    bool apply(Vec &&q, real_t γ);

    /// Apply the inverse Hessian approximation to the given vector q, applying
    /// only the columns and rows of the Hessian in the index set J.
    template <class Vec, class IndexVec>
    bool apply(Vec &&q, real_t γ, const IndexVec &J);

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
    length_t n() const { return sto.rows() - 1; }
    /// Get the number of previous vectors s and y stored in the buffer.
    length_t history() const { return sto.cols() / 2; }
    /// Get the next index in the circular buffer of previous s and y vectors.
    index_t succ(index_t i) const { return i + 1 < history() ? i + 1 : 0; }

    auto s(index_t i) { return sto.col(2 * i).topRows(n()); }
    auto s(index_t i) const { return sto.col(2 * i).topRows(n()); }
    auto y(index_t i) { return sto.col(2 * i + 1).topRows(n()); }
    auto y(index_t i) const { return sto.col(2 * i + 1).topRows(n()); }
    real_t &ρ(index_t i) { return sto.coeffRef(n(), 2 * i); }
    const real_t &ρ(index_t i) const { return sto.coeff(n(), 2 * i); }
    real_t &α(index_t i) { return sto.coeffRef(n(), 2 * i + 1); }
    const real_t &α(index_t i) const { return sto.coeff(n(), 2 * i + 1); }

  private:
    using storage_t = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic>;

    storage_t sto;
    index_t idx = 0;
    bool full  = false;
    Params params;
};

} // namespace quala