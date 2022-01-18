import quala as qa
import numpy as np
from typing import List

A = np.array([[20, -10], [-10, 30]], dtype=np.float64)
b = np.array([10, 20], dtype=np.float64)
n = A.shape[1]

ε = 1e-15


def test_anderson():
    x = b.copy()
    aa = qa.AndersonAccel({'memory': n}, n)
    res_aa: List[float] = []
    for i in range(1 + n + 2):
        r = A @ x - b
        g = r + x
        res_aa.append(np.linalg.norm(r))
        print(f"i: {i}")
        print(f"x:    {x}")
        print(f"g:    {g}")
        print(f"r:    {r}")
        if i == 0:
            aa.initialize(g, r)
            x = g
        else:
            x = aa.compute(g, r)

    print(f"i: final")
    print(f"x:    {x}")
    print(f"r:    {r}")
    assert np.allclose(x, [1, 1], rtol=ε, atol=ε)


def test_lbfgs():
    x = b.copy()
    r = A @ x - b
    lbfgs = qa.LBFGS({'memory': 2 * n}, n)
    res_lbfgs: List[float] = []
    for i in range(1 + 2 * n + 2):
        res_lbfgs.append(np.linalg.norm(r))
        print(f"i: {i}")
        print(f"x:    {x}")
        print(f"r:    {r}")
        q = r.copy()
        lbfgs.apply(q, -1)
        x_new = x - q
        r_new = A @ x_new - b
        lbfgs.update(x, x_new, r, r_new, qa.LBFGS.Sign.Positive)
        x = x_new
        r = r_new

    print(f"i: final")
    print(f"x:    {x}")
    print(f"r:    {r}")
    assert np.allclose(x, [1, 1], rtol=ε, atol=ε)


def test_broyden_good():
    x = b.copy()
    r = A @ x - b
    broyden = qa.BroydenGood({'memory': 3 * n}, n)
    res_broyden: List[float] = []
    for i in range(1 + 2 * n + 2):
        res_broyden.append(np.linalg.norm(r))
        print(f"i: {i}")
        print(f"x:    {x}")
        print(f"r:    {r}")
        q = r.copy()
        broyden.apply(q)
        x_new = x - q
        r_new = A @ x_new - b
        broyden.update(x, x_new, r, r_new)
        x = x_new
        r = r_new

    print(f"i: final")
    print(f"x:    {x}")
    print(f"r:    {r}")
    assert np.allclose(x, [1, 1], rtol=ε, atol=ε)
