from typing import List
import casadi as cs
import numpy.random as nprand
import numpy.linalg as la
import numpy as np
import quala as qa
import os

n = 10  # problem dimension
ε = 1e-12  # assertion tolerance

# Positive definite “QP” with a twist (including a sin for nonlinearity)
rng = nprand.default_rng(seed=123)
A = rng.random(size=(n, n))
Q, _ = la.qr(A)
D = np.diag(rng.normal(scale=4, size=(n, )))
A = Q.T @ D.T @ D @ Q
q = rng.normal(scale=1, size=(n, ))
print(f"{la.cond(A)=}")
x_ = cs.SX.sym('x', n)
px = x_ + 0.2 * cs.sin(x_[::-1])
f_ = 0.5 * (x_.T @ A @ px) + cs.dot(q, x_)
f = cs.Function("f", [x_], [f_])
grad_f_ = cs.gradient(f_, x_)
grad_f = cs.Function("grad_f", [x_], [grad_f_])
hess_f_ = cs.jacobian(grad_f_, x_)
hess_f = cs.Function("hess_f", [x_], [hess_f_])
L = la.norm(D)**2
# Uncomment the next line for gradient descent instead of Newton:
# hess_f = lambda _: L * cs.DM.eye(n)

# solve r(x) = -(H(x))⁻¹ ∇f(x) = 0 or g(x) = r(x) + x = x


def test_anderson():
    x = np.ones((n, 1))
    m = n - 2
    print(f"{m=}")
    aa = qa.AndersonAccel({'memory': m}, n)

    Gk = np.zeros((n, 0))
    Rk = np.zeros((n, 0))
    r_prev = np.nan * np.ones((n, 1))

    p_stack = cs.DM.zeros(n, 0)

    res_aa: List[float] = []
    x_aa_diff_quala: List[float] = []
    x_aa_diff_david: List[float] = []
    for i in range(3 * n):
        r = -la.solve(hess_f(x).full(), grad_f(x).full())  # Newton step (res)
        g = r + x
        res_aa.append(la.norm(r))
        print(f"{i}: {res_aa[-1]}")

        # Initialize
        if i == 0:
            # quala implementation
            aa.initialize(g, r)

            # Python implementation
            r_prev = r.copy()
            Gk = g.copy()

            # David's implementation
            x_stack = x.copy()
            p_stack = r.copy()

            x = g.copy()
        # Anderson update
        else:
            mk = min(i, m)
            print(f"    {mk=}")

            # quala implementation
            x_aa_quala = aa.compute(g, r)
            x_aa_quala = np.array([x_aa_quala]).T

            # Python implementation
            Rk = np.hstack((Rk, r - r_prev))
            Gk = np.hstack((Gk, g))
            γ_LS, _, _, _ = la.lstsq(Rk[:, -mk:], r, rcond=None)
            α_LS = np.nan * np.ones((mk + 1, ))
            α_LS[0] = γ_LS[0]
            for j in range(1, mk):
                α_LS[j] = γ_LS[j] - γ_LS[j - 1]
            α_LS[mk] = 1 - γ_LS[mk - 1]
            x_aa_py = np.zeros((n, 1))
            for j in range(0, mk + 1):
                gj = Gk[:, [i - mk + j]]
                x_aa_py += α_LS[j] * gj
            r_prev = r.copy()

            # David's implementation
            x_tmp = x.copy()  # <?>  (current iterate)
            p_tmp = r.copy()  # <?>  (current residual, Newton step)
            if i <= m:  # Fixed
                print("    mk <= m")
                p_stack = cs.horzcat(p_tmp, p_stack)
                x_stack = cs.horzcat(x_tmp, x_stack)
            else:
                print("    mk > m")
                p_stack = cs.horzcat(p_tmp, p_stack[:, 0:-1])
                x_stack = cs.horzcat(x_tmp, x_stack[:, 0:-1])

            F_k = p_stack[:, 0:-1] - p_stack[:, 1:]
            E_k = x_stack[:, 0:-1] - x_stack[:, 1:]

            pinv_Fk = np.linalg.pinv(F_k)
            gamma_k = pinv_Fk @ p_tmp

            x_aa_david = x_stack[:, 0] + p_tmp - (E_k + F_k) @ gamma_k

            # Compare results
            rel_diff = lambda x, y: la.norm(x - y) / la.norm(x)
            x_aa_diff_quala.append(rel_diff(x_aa_py, x_aa_quala))
            print(f"    {x_aa_diff_quala[-1]=}")
            assert x_aa_diff_quala[-1] < ε, f"Failed on iteration {i}"
            x_aa_diff_david.append(rel_diff(x_aa_py, x_aa_david))
            print(f"    {x_aa_diff_david[-1]=}")
            assert x_aa_diff_david[-1] < ε, f"Failed on iteration {i}"

            # Next iterate
            x = x_aa_quala.copy()

    if not "PYTEST_CURRENT_TEST" in os.environ:
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 14,
            "lines.linewidth": 1,
        })

        plt.figure()
        plt.title("Residual")
        plt.semilogy(res_aa, '.-')
        plt.ylim([1e-16, None])
        plt.xlabel("Iteration $k$")
        plt.ylabel(r"$\|r(x^k)\|$")
        plt.tight_layout()

        plt.figure()
        plt.title("Difference between iterations")
        plt.semilogy(x_aa_diff_quala, '.-', label="Python vs quala")
        plt.semilogy(x_aa_diff_david, '.-', label="Python vs David")
        plt.legend()
        plt.xlabel("Iteration $k$")
        plt.ylabel(r"Rel. difference $\|x^a_{k+1}-x^b_{k+1}\|/\|x^a_{k+1}\|$")
        plt.tight_layout()
        plt.show()

    r = -la.solve(hess_f(x).full(), grad_f(x).full())  # Newton step (res)
    print(f"final: {la.norm(r)}")
    print(f"x:\r\n{x}")
    print(f"r:\r\n{r}")
    assert np.allclose(r, np.zeros((n, )), rtol=ε, atol=ε)


if __name__ == '__main__':
    test_anderson()
