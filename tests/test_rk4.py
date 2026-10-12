"""The integrator must have the convergence order the theory promises.

Global error of Euler is O(h), of RK4 is O(h^4). Halving h should divide the
error by 2 and 16 respectively. These tests measure the exponent directly.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from sir.model import SIRParams, sir_rhs
from sir.rk4 import euler_step, integrate, rk4_step

def decay(t, y, params):
    """y' = -y, exact solution y = exp(-t)."""
    return -y

def cosine(t, y, params):
    """y' = cos t, exact solution y = sin t."""
    return np.array([np.cos(t)])

def error_at_end(step, h):
    t, Y = integrate(decay, np.array([1.0]), 0.0, 2.0, h, None, step=step)
    return abs(Y[-1, 0] - np.exp(-2.0))

@pytest.mark.parametrize("step, expected_order", [(euler_step, 1.0), (rk4_step, 4.0)])
def test_measured_convergence_order(step, expected_order):
    hs = [0.1, 0.05, 0.025]
    errors = [error_at_end(step, h) for h in hs]
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(len(hs) - 1)]
    for order in orders:
        assert abs(order - expected_order) < 0.1, (orders, errors)

def test_rk4_on_cosine_matches_sine():
    t, Y = integrate(cosine, np.array([0.0]), 0.0, 2 * np.pi, 0.01, None)
    assert np.max(np.abs(Y[:, 0] - np.sin(t))) < 1e-8

def test_rk4_matches_scipy_on_sir():
    params = SIRParams(beta=0.5, gamma=0.25, N=1000.0)
    y0 = np.array([999.0, 1.0, 0.0])
    t, Y = integrate(sir_rhs, y0, 0.0, 60.0, 0.01, params)
    ref = solve_ivp(
        lambda tt, yy: sir_rhs(tt, yy, params), (0.0, 60.0), y0,
        method="RK45", rtol=1e-10, atol=1e-12, dense_output=True,
    )
    assert np.max(np.abs(Y - ref.sol(t).T)) < 1e-6