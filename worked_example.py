"""One RK4 step of the SIR model, small enough to check with a calculator.

N = 1000, one infected person, beta = 0.5, gamma = 0.25 (so R0 = 2), h = 1 day.

Run from this folder:  python worked_example.py
"""

import numpy as np

from sir.model import SIRParams, r0, sir_rhs, solve_final_size
from sir.rk4 import euler_step, integrate, rk4_step

np.set_printoptions(precision=4, suppress=True)

params = SIRParams(beta=0.5, gamma=0.25, N=1000.0)
y0 = np.array([999.0, 1.0, 0.0])
h = 1.0

print(f"R0 = beta / gamma = {params.beta} / {params.gamma} = {r0(params)}")
print(f"y0 = [S, I, R] = {y0}\n")

print("--- the four RK4 slopes ---")
k1 = sir_rhs(0.0, y0, params)
print(f"k1 = f(y0)              = {k1}")
y_mid1 = y0 + 0.5 * h * k1
k2 = sir_rhs(0.5 * h, y_mid1, params)
print(f"  y0 + h/2 k1         = {y_mid1}")
print(f"k2 = f(y0 + h/2 k1)     = {k2}")
y_mid2 = y0 + 0.5 * h * k2
k3 = sir_rhs(0.5 * h, y_mid2, params)
print(f"  y0 + h/2 k2         = {y_mid2}")
print(f"k3 = f(y0 + h/2 k2)     = {k3}")
y_end = y0 + h * k3
k4 = sir_rhs(h, y_end, params)
print(f"  y0 + h k3           = {y_end}")
print(f"k4 = f(y0 + h k3)       = {k4}")
weighted = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
print(f"(k1 + 2k2 + 2k3 + k4) / 6  = {weighted}\n")

print("--- one step, three ways ---")
y1_rk4 = rk4_step(sir_rhs, 0.0, y0, h, params)
y1_euler = euler_step(sir_rhs, 0.0, y0, h, params)
t_fine, Y_fine = integrate(sir_rhs, y0, 0.0, h, 1e-4, params)
print(f"RK4  (h = 1)          = {y1_rk4}")
print(f"Euler (h = 1)         = {y1_euler}")
print(f"reference (h = 1e-4)  = {Y_fine[-1]}")
print(f"RK4 error in I        = {abs(y1_rk4[1] - Y_fine[-1, 1]):.2e}")
print(f"Euler error in I      = {abs(y1_euler[1] - Y_fine[-1, 1]):.2e}\n")

print("--- how big does the outbreak get? ---")
s_inf = solve_final_size(params, y0[0])
print(f"solve ln(999 / S_inf) = 2 (1 - S_inf / 1000) by bisection")
print(f"S_inf = {s_inf:.2f}  ->  {params.N - s_inf:.2f} of 1000 people get infected "
      f"({(params.N - s_inf) / params.N:.1%})")
t, Y = integrate(sir_rhs, y0, 0.0, 400.0, 0.1, params)
print(f"RK4 run to day 400 gives S = {Y[-1, 0]:.2f}, peak I = {Y[:, 1].max():.1f} "
      f"on day {t[np.argmax(Y[:, 1])]:.1f} where S = {Y[np.argmax(Y[:, 1]), 0]:.1f} "
      f"(N / R0 = {params.N / r0(params):.0f})")