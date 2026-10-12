"""The hand-written Nelder-Mead must find known minima."""

import numpy as np

from sir.fit import fit_sir, nelder_mead
from sir.model import SIRParams, sir_rhs
from sir.rk4 import integrate

def test_nelder_mead_finds_minimum_of_quadratic():
    x, fx, it = nelder_mead(lambda p: (p[0] - 1.0) ** 2 + (p[1] - 2.0) ** 2, np.array([5.0, -3.0]))
    assert np.allclose(x, [1.0, 2.0], atol=1e-6)
    assert fx < 1e-12

def test_nelder_mead_finds_minimum_of_rosenbrock():
    """The banana valley: gradient-free methods must follow a curved trough."""
    rosen = lambda p: (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2
    x, fx, it = nelder_mead(rosen, np.array([-1.2, 1.0]), max_iter=5000)
    assert np.allclose(x, [1.0, 1.0], atol=1e-4)

def test_fit_recovers_known_parameters_from_noise_free_data():
    true = SIRParams(beta=1.5, gamma=0.5, N=1000.0)
    t_data = np.arange(0.0, 21.0)
    t, Y = integrate(sir_rhs, np.array([999.0, 1.0, 0.0]), 0.0, 20.0, 0.05, true)
    I_data = np.interp(t_data, t, Y[:, 1])
    result = fit_sir(t_data, I_data, N=1000.0, I0=1.0, initial_guess=(1.0, 0.3))
    assert abs(result.beta - true.beta) < 1e-3
    assert abs(result.gamma - true.gamma) < 1e-3
    assert result.sse < 1e-6