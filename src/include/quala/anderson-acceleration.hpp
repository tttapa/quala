#pragma once

#include <quala/detail/anderson-helpers.hpp>
#include <stdexcept>

namespace quala {

/// Parameters for the @ref AndersonAccel class.
struct AndersonAccelParams {
    /// Length of the history to keep (the number of columns in the QR
    /// factorization).
    /// If this number is greater than the problem dimension, the memory is set
    /// to the problem dimension (otherwise the system is underdetermined).
    length_t memory = 10;
};

/// Anderson Acceleration.
///
/// @todo   Condition estimation of the QR factorization.
///
/// @ingroup accelerators-grp
class AndersonAccel {
  public:
    using Params = AndersonAccelParams;

    /// @param  params
    ///         Parameters.
    AndersonAccel(Params params) : params(params) {}
    /// @param  params
    ///         Parameters.
    /// @param  n
    ///         Problem dimension (size of the vectors).
    AndersonAccel(Params params, length_t n) : params(params) { resize(n); }

    /// Change the problem dimension. Flushes the history.
    /// @param  n
    ///         Problem dimension (size of the vectors).
    void resize(length_t n) {
        length_t m_AA = std::min(n, params.memory); // TODO: support m > n?
        qr.resize(n, m_AA);
        G.resize(n, m_AA);
        rₖ₋₁.resize(n);
        γ_LS.resize(m_AA);
        initialized = false;
    }

    /// Call this function on the first iteration to initialize the accelerator.
    void initialize(crvec g₀, vec r₀) {
        assert(g₀.size() == vec::Index(n()));
        assert(r₀.size() == vec::Index(n()));
        G.col(0) = g₀;
        rₖ₋₁     = std::move(r₀);
        qr.reset();
        initialized = true;
    }

    /// Compute the accelerated iterate @f$ x^k_\text{AA} @f$, given the
    /// function value at the current iterate @f$ g^k = g(x^k) @f$ and the
    /// corresponding residual @f$ r^k = g^k - x^k @f$.
    void compute(crvec gₖ, crvec rₖ, rvec xₖ_aa) {
        if (!initialized)
            throw std::logic_error("AndersonAccel::compute() called before "
                                   "AndersonAccel::initialize()");
        minimize_update_anderson(qr, G,        // inout
                                 rₖ, rₖ₋₁, gₖ, // in
                                 γ_LS, xₖ_aa); // out
        rₖ₋₁ = rₖ;
    }
    /// @copydoc compute(crvec, crvec, rvec)
    void compute(crvec gₖ, vec &&rₖ, rvec xₖ_aa) {
        if (!initialized)
            throw std::logic_error("AndersonAccel::compute() called before "
                                   "AndersonAccel::initialize()");
        minimize_update_anderson(qr, G,        // inout
                                 rₖ, rₖ₋₁, gₖ, // in
                                 γ_LS, xₖ_aa); // out
        rₖ₋₁ = std::move(rₖ);
    }

    /// Reset the accelerator (but keep the last function value and residual, so
    /// calling @ref initialize is not necessary).
    void reset() {
        index_t newest_g_idx = qr.ring_tail();
        if (newest_g_idx != 0)
            G.col(0) = G.col(newest_g_idx);
        qr.reset();
    }

    /// Get the problem dimension.
    length_t n() const { return qr.n(); }
    /// Get the maximum number of stored columns.
    length_t history() const { return qr.m(); }
    /// Get the number of columns currently stored in the buffer.
    length_t current_history() const { return qr.current_history(); }

    /// Get the parameters.
    const Params &get_params() const { return params; }

  private:
    Params params;
    LimitedMemoryQR qr;
    mat G;
    vec rₖ₋₁;
    vec γ_LS;
    bool initialized = false;
};

} // namespace quala