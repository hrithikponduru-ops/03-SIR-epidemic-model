"""The SIR compartment model and its closed-form consequences.

A population of N people is split into Susceptible, Infected and Recovered.
Infection needs a susceptible to meet an infected person, so its rate is
proportional to S * I. Recovery empties I at a constant per-person rate.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SIRParams:
    beta: float  # contacts that transmit, per person per day
    gamma: float # recoveries per infected person per day (1 / infectious period)
    N: float     # total population, constant


def sir_rhs(t: float, y: np.ndarray, params: SIRParams) -> np.ndarray:
    """dy/dt for y = [S, I, R]. The three entries sum to zero."""
    S, I, R = y
    new_infections = params.beta * S * I / params.N
    recoveries = params.gamma * I
    return np.array([-new_infections, new_infections - recoveries, recoveries])


def r0(params: SIRParams) -> float:
    """Basic reproduction number: infections caused by one case in a fully
    susceptible population, beta contacts/day for 1/gamma days."""
    return params.beta / params.gamma


def final_size_residual(s_inf: float, s0: float, r0_value: float, N: float) -> float:
    """Zero exactly when s_inf satisfies ln(S0 / S_inf) = R0 (1 - S_inf / N).

    Derived by dividing dS/dt by dR/dt, integrating, and using I -> 0 at the
    end of the outbreak (so R_inf = N - S_inf when R(0) = 0).
    """
    return np.log(s0 / s_inf) - r0_value * (1.0 - s_inf / N)


def bisect(f, lo: float, hi: float, tol: float = 1e-10, max_iter: int = 200) -> float:
    """Root of f on [lo, hi] where f(lo) and f(hi) have opposite signs.

    Each step halves the bracket, so the error after k steps is (hi - lo) / 2^k.
    """
    f_lo = f(lo)
    if f_lo * f(hi) > 0:
        raise ValueError("bisect needs a sign change on [lo, hi]")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if f_mid == 0.0 or hi - lo < tol:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def solve_final_size(params: SIRParams, s0: float) -> float:
    """S_inf: the number of people who never get infected, from the implicit
    final-size relation. The root lies strictly between 0 and s0."""
    rho = r0(params)
    return bisect(lambda s: final_size_residual(s, s0, rho, params.N), 1e-9 * s0, s0 * (1 - 1e-12))