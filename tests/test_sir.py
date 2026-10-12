"""Each test encodes a theorem about the SIR model, not just a behaviour."""

import numpy as np

from sir.model import SIRParams, r0, sir_rhs, solve_final_size
from sir.rk4 import integrate

PARAMS = SIRParams(beta=0.5, gamma=0.25, N=1000.0)  # R0 = 2
Y0 = np.array([999.0, 1.0, 0.0])

def test_population_is_conserved():
    """The three right-hand sides sum to zero, so S + I + R never changes."""
    t, Y = integrate(sir_rhs, Y0, 0.0, 60.0, 0.1, PARAMS)
    assert np.max(np.abs(Y.sum(axis=1) - PARAMS.N)) < 1e-9

def test_compartments_stay_non_negative():
    t, Y = integrate(sir_rhs, Y0, 0.0, 200.0, 0.1, PARAMS)
    assert np.all(Y >= 0.0)

def test_final_size_relation():
    """ln(S0 / S_inf) = R0 (1 - S_inf / N): exact, from dS/dR = -R0 S / N."""
    t, Y = integrate(sir_rhs, Y0, 0.0, 400.0, 0.1, PARAMS)
    s_inf = Y[-1, 0]
    residual = np.log(Y0[0] / s_inf) - r0(PARAMS) * (1.0 - s_inf / PARAMS.N)
    assert abs(residual) < 1e-4
    assert abs(s_inf - solve_final_size(PARAMS, Y0[0])) < 1e-3

def test_infection_peaks_when_s_equals_n_over_r0():
    """dI/dt = I (beta S / N - gamma) = 0 <=> S = N / R0."""
    h = 0.01
    t, Y = integrate(sir_rhs, Y0, 0.0, 60.0, h, PARAMS)
    peak = int(np.argmax(Y[:, 1]))
    s_at_peak = Y[peak, 0]
    max_ds_per_step = h * np.max(np.abs(np.diff(Y[:, 0]))) / h
    assert abs(s_at_peak - PARAMS.N / r0(PARAMS)) < max_ds_per_step

def test_no_outbreak_when_r0_below_one():
    """With S <= N and R0 < 1, dI/dt = I (beta S / N - gamma) < 0 always."""
    params = SIRParams(beta=0.2, gamma=0.25, N=1000.0)
    t, Y = integrate(sir_rhs, Y0, 0.0, 60.0, 0.1, params)
    assert np.all(np.diff(Y[:, 1]) <= 0.0)