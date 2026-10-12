"""Fixed-step integrators for y' = f(t, y, params).

Euler follows the slope at the start of the step. RK4 samples the slope four
times (start, two midpoints, end) and averages them with Simpson's weights
1, 2, 2, 1. That single change moves the global error from O(h) to O(h^4).
"""

import numpy as np


def euler_step(f, t: float, y: np.ndarray, h: float, params) -> np.ndarray:
    """y_{n+1} = y_n + h f(t_n, y_n). Local error O(h^2), global O(h)."""
    return y + h * f(t, y, params)


def rk4_step(f, t: float, y: np.ndarray, h: float, params) -> np.ndarray:
    """Classical fourth-order Runge-Kutta. Local error O(h^5), global O(h^4)."""
    k1 = f(t, y, params)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1, params)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2, params)
    k4 = f(t + h, y + h * k3, params)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate(f, y0: np.ndarray, t0: float, t1: float, h: float, params, step=rk4_step):
    """March from t0 to t1 in steps of h. Returns (t, Y) with Y[i] = y(t[i]).

    The number of steps is rounded so the grid lands on t1 exactly; h is
    adjusted by at most one part in n to make that happen.
    """
    n = max(1, int(round((t1 - t0) / h)))
    h_exact = (t1 - t0) / n
    t = t0 + h_exact * np.arange(n + 1)
    Y = np.empty((n + 1, len(y0)))
    Y[0] = y0
    for i in range(n):
        Y[i + 1] = step(f, t[i], Y[i], h_exact, params)
    return t, Y