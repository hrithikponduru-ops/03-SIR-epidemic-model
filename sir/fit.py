"""Fit beta and gamma to observed case counts by least squares.

The loss (sum of squared errors between the model's I(t) and the data) has
no usable gradient: evaluating it means running an ODE solver. Nelder-Mead
needs only function values. It keeps a simplex of d + 1 points (a triangle
in 2-D) and moves the worst vertex through the centroid of the others.
"""

from dataclasses import dataclass

import numpy as np

from sir.model import SIRParams, sir_rhs
from sir.rk4 import integrate

REFLECT, EXPAND, CONTRACT, SHRINK = 1.0, 2.0, 0.5, 0.5


@dataclass(frozen=True)
class FitResult:
    beta: float
    gamma: float
    sse: float
    iterations: int


def _initial_simplex(x0: np.ndarray, step: float) -> np.ndarray:
    """x0 plus one vertex per coordinate, displaced by a relative step."""
    vertices = [x0]
    for i in range(len(x0)):
        v = x0.copy()
        v[i] = v[i] * (1.0 + step) if v[i] != 0.0 else step
        vertices.append(v)
    return np.array(vertices)


def _try_replace_worst(f, simplex: np.ndarray, values: np.ndarray):
    """Reflection, expansion or contraction of the worst vertex.

    Returns (new_vertex, new_value) or None if all three failed and the
    simplex must shrink instead. simplex must be sorted best to worst.
    """
    centroid = simplex[:-1].mean(axis=0)
    worst, f_worst = simplex[-1], values[-1]
    xr = centroid + REFLECT * (centroid - worst)
    fr = f(xr)
    if values[0] <= fr < values[-2]:
        return xr, fr
    if fr < values[0]:
        xe = centroid + EXPAND * (xr - centroid)
        fe = f(xe)
        return (xe, fe) if fe < fr else (xr, fr)
    if fr < f_worst:
        xc = centroid + CONTRACT * (xr - centroid)  # outside contraction
    else:
        xc = centroid + CONTRACT * (worst - centroid)  # inside contraction
    fc = f(xc)
    if fc < min(fr, f_worst):
        return xc, fc
    return None


def nelder_mead(f, x0, step=0.1, xtol=1e-8, ftol=1e-10, max_iter=2000):
    """Minimise f without derivatives. Returns (x_best, f_best, iterations)."""
    simplex = _initial_simplex(np.asarray(x0, dtype=float), step)
    values = np.array([f(v) for v in simplex])
    it = 0
    for it in range(1, max_iter + 1):
        order = np.argsort(values)
        simplex, values = simplex[order], values[order]
        size = np.max(np.linalg.norm(simplex[1:] - simplex[0], axis=1))
        if size < xtol and values[-1] - values[0] < ftol:
            break
        replacement = _try_replace_worst(f, simplex, values)
        if replacement is not None:
            simplex = np.vstack([simplex[:-1], replacement[0]])
            values = np.append(values[:-1], replacement[1])
        else:
            simplex = simplex[0] + SHRINK * (simplex - simplex[0])
            values = np.array([f(v) for v in simplex])
    best = int(np.argmin(values))
    return simplex[best], float(values[best]), it


def sse(theta, t_data, I_data, N: float, I0: float, h: float) -> float:
    """Sum of squared errors between the model's I(t) and the observations."""
    beta, gamma = theta
    if beta <= 0.0 or gamma <= 0.0:
        return np.inf
    params = SIRParams(beta=beta, gamma=gamma, N=N)
    y0 = np.array([N - I0, I0, 0.0])
    t, Y = integrate(sir_rhs, y0, 0.0, float(t_data[-1]), h, params)
    return float(np.sum((np.interp(t_data, t, Y[:, 1]) - I_data) ** 2))


def fit_sir(t_data, I_data, N: float, I0: float, initial_guess, h: float = 0.05) -> FitResult:
    """Least-squares (beta, gamma) from infected counts at times t_data.

    Integration starts at t = 0 with I(0) = I0 and R(0) = 0.
    """
    t_data, I_data = np.asarray(t_data, float), np.asarray(I_data, float)
    theta, loss, it = nelder_mead(lambda th: sse(th, t_data, I_data, N, I0, h), initial_guess)
    return FitResult(beta=float(theta[0]), gamma=float(theta[1]), sse=loss, iterations=it)