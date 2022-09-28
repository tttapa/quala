#pragma once

#include <quala/decl/lbfgs.hpp>
#include <stdexcept>
#include <type_traits>

namespace quala {

inline bool LBFGS::update_valid(const LBFGSParams &params, real_t yᵀs,
                                real_t sᵀs, real_t pᵀp) {
    // Check if this L-BFGS update is accepted
    if (sᵀs <= params.min_abs_s)
        return false;
    if (not std::isfinite(yᵀs))
        return false;
    real_t a_yᵀs = params.force_pos_def ? yᵀs : std::abs(yᵀs);
    if (a_yᵀs <= params.min_div_fac * sᵀs)
        return false;

    // CBFGS condition: https://epubs.siam.org/doi/10.1137/S1052623499354242
    if (params.cbfgs) {
        const real_t α = params.cbfgs.α;
        const real_t ϵ = params.cbfgs.ϵ;
        // Condition: yᵀs / sᵀs >= ϵ ‖p‖^α
        bool cbfgs_cond = a_yᵀs >= sᵀs * ϵ * std::pow(pᵀp, α / 2);
        if (not cbfgs_cond)
            return false;
    }

    return true;
}

template <class VecS, class VecY>
inline bool LBFGS::update_sy(const anymat<VecS> &s, const anymat<VecY> &y,
                             real_t pₙₑₓₜᵀpₙₑₓₜ, bool forced) {
    real_t yᵀs = y.dot(s);
    real_t ρ   = 1 / yᵀs;
    if (not forced) {
        real_t sᵀs = s.squaredNorm();
        if (not update_valid(params, yᵀs, sᵀs, pₙₑₓₜᵀpₙₑₓₜ))
            return false;
    }

    // Store the new s and y vectors
    sto.s(idx) = s;
    sto.y(idx) = y;
    sto.ρ(idx) = ρ;

    // Increment the index in the circular buffer
    idx = succ(idx);
    full |= idx == 0;

    return true;
}

inline bool LBFGS::update(crvec xₖ, crvec xₙₑₓₜ, crvec pₖ, crvec pₙₑₓₜ, Sign sign,
                          bool forced) {
    const auto s = xₙₑₓₜ - xₖ;
    const auto y = (sign == Sign::Positive) ? pₙₑₓₜ - pₖ : pₖ - pₙₑₓₜ;
    real_t pₙₑₓₜᵀpₙₑₓₜ = params.cbfgs ? pₙₑₓₜ.squaredNorm() : 0;
    return update_sy(s, y, pₙₑₓₜᵀpₙₑₓₜ, forced);
}

inline bool LBFGS::apply(rvec q, real_t γ) {
    // Only apply if we have previous vectors s and y
    if (idx == 0 && not full)
        return false;

    // If the step size is negative, compute it as sᵀy/yᵀy
    if (γ < 0) {
        auto new_idx = pred(idx);
        real_t yᵀy   = y(new_idx).squaredNorm();
        γ            = 1 / (ρ(new_idx) * yᵀy);
    }

    foreach_rev([&](index_t i) {
        α(i) = ρ(i) * s(i).dot(q);
        q -= α(i) * y(i);
    });

    // r ← H₀ q
    q *= γ;

    foreach_fwd([&](index_t i) {
        real_t β = ρ(i) * y(i).dot(q);
        q -= (β - α(i)) * s(i);
    });

    return true;
}

template <class IndexVec>
bool LBFGS::apply(rvec q, real_t γ, const IndexVec &J) {
    // Only apply if we have previous vectors s and y
    if (idx == 0 && not full)
        return false;
    const bool fullJ = q.size() == static_cast<index_t>(J.size());

    if (params.cbfgs)
        throw std::invalid_argument("CBFGS check not supported when using "
                                    "masked version of LBFGS::apply()");

    // Eigen 3.3.9 doesn't yet support indexing using a vector of indices
    // so we'll have to do it manually.
    // TODO: Abstract this away in an expression template / nullary expression?
    //       Or wait for Eigen update?
    // Update: Eigen 3.4's indexing seems significantly slower, so the manual
    //         for loops stay for now.

    // Dot product of two vectors, adding only the indices in set J
    const auto dotJ = [&J, fullJ](const auto &a, const auto &b) {
        if (fullJ) {
            return a.dot(b);
        } else {
            real_t acc = 0;
            for (auto j : J)
                acc += a(j) * b(j);
            return acc;
        }
    };
    // y -= a x, scaling and subtracting only the indices in set J
    const auto axmyJ = [&J, fullJ](real_t a, const auto &x, auto &y) {
        if (fullJ) {
            y -= a * x;
        } else {
            for (auto j : J)
                y(j) -= a * x(j);
        }
    };
    // x *= a, scaling only the indices in set J
    const auto scalJ = [&J, fullJ](real_t a, auto &x) {
        if (fullJ) {
            x *= a;
        } else {
            for (auto j : J)
                x(j) *= a;
        }
    };

    foreach_rev([&](index_t i) {
        // Recompute ρ, it depends on the index set J. Note that even if ρ was
        // positive for the full vectors s and y, that's not necessarily the
        // case for the smaller vectors s(J) and y(J).
        real_t yᵀs = dotJ(s(i), y(i));
        real_t sᵀs = dotJ(s(i), s(i));
        ρ(i)       = 1 / yᵀs;
        // Check if we should include this pair of vectors
        if (not update_valid(params, yᵀs, sᵀs, 0)) {
            ρ(i) = NaN;
            return;
        }

        α(i) = ρ(i) * dotJ(s(i), q); // αᵢ = ρᵢ〈sᵢ, q〉
        axmyJ(α(i), y(i), q);        // q -= αᵢ yᵢ

        if (γ < 0) {
            // Compute step size based on most recent valid yᵀs/yᵀy
            real_t yᵀy = dotJ(y(i), y(i));
            γ          = 1 / (ρ(i) * yᵀy);
        }
    });

    // If all ρ == 0, fail
    if (γ < 0)
        return false;

    // r ← H₀ q
    scalJ(γ, q); // q *= γ

    foreach_fwd([&](index_t i) {
        if (std::isnan(ρ(i)))
            return;
        real_t β = ρ(i) * dotJ(y(i), q); // βᵢ = ρᵢ〈yᵢ, q〉
        axmyJ(β - α(i), s(i), q);        // q -= (βᵢ - αᵢ) sᵢ
    });

    return true;
}

inline void LBFGS::reset() {
    idx  = 0;
    full = false;
}

inline void LBFGS::resize(length_t n) {
    if (params.memory < 1)
        throw std::invalid_argument("LBFGS::Params::memory must be >= 1");
    sto.resize(n, params.memory);
    reset();
}

inline void LBFGSStorage::resize(length_t n, length_t history) {
    sto.resize(n + 1, history * 2);
}

inline void LBFGS::scale_y(real_t factor) {
    if (full) {
        for (index_t i = 0; i < history(); ++i) {
            y(i) *= factor;
            ρ(i) *= 1 / factor;
        }
    } else {
        for (index_t i = 0; i < idx; ++i) {
            y(i) *= factor;
            ρ(i) *= 1 / factor;
        }
    }
}

} // namespace quala