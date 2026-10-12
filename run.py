"""Fit the SIR model to the 1978 English boarding-school influenza outbreak.

Data: Anonymous, "Influenza in a boarding school", BMJ, 4 March 1978, p. 587.
763 boys; the table gives the number confined to bed on each of 14 days.

Run from this folder: python run.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sir.fit import fit_sir
from sir.model import SIRParams, r0, sir_rhs, solve_final_size
from sir.rk4 import integrate

N = 763.0
I0 = 1.0
OBSERVED_TOTAL_IN_BED = 512  # reported in the BMJ letter
INITIAL_GUESS = (2.0, 0.5)
H = 0.05
COUNTERFACTUAL_BETA_FACTOR = 0.8

INK, BLUE, GREEN, RED, GREY, AMBER = "#1f2937", "#2563eb", "#059669", "#dc2626", "#9ca3af", "#d97706"

def load_data():
    table = np.genfromtxt("data/boarding_school_1978.csv", delimiter=",", skip_header=1)
    return table[:, 0], table[:, 1]

def simulate(params, t_end):
    return integrate(sir_rhs, np.array([N - I0, I0, 0.0]), 0.0, t_end, H, params)

def describe(label, params, t, Y):
    peak = int(np.argmax(Y[:, 1]))
    s_inf = solve_final_size(params, N - I0)
    print(f"{label}: R0 = {r0(params):.2f}, peak I = {Y[peak, 1]:.0f} on day {t[peak]:.1f}, "
          f"final size = {N - s_inf:.0f} of {N:.0f} ({(N - s_inf) / N:.1%})")
    return t[peak], Y[peak, 1], N - s_inf

def main():
    days, in_bed = load_data()
    
    fit = fit_sir(days, in_bed, N, I0, INITIAL_GUESS, h=H)
    params = SIRParams(beta=fit.beta, gamma=fit.gamma, N=N)
    print(f"fitted beta = {fit.beta:.4f} /day, gamma = {fit.gamma:.4f} /day "
          f"(infectious period {1 / fit.gamma:.2f} days), R0 = {r0(params):.3f}")
    print(f"SSE = {fit.sse:.1f} over {len(days)} days, Nelder-Mead iterations = {fit.iterations}")
    obs_peak = int(np.argmax(in_bed))
    print(f"observed peak: {in_bed[obs_peak]:.0f} boys on day {days[obs_peak]:.0f}")

    t, Y = simulate(params, 30.0)
    describe("fitted model", params, t, Y)
    print(f"observed total confined to bed: {OBSERVED_TOTAL_IN_BED} of {N:.0f} "
          f"({OBSERVED_TOTAL_IN_BED / N:.1%})")

    cf_params = SIRParams(beta=fit.beta * COUNTERFACTUAL_BETA_FACTOR, gamma=fit.gamma, N=N)
    t_cf, Y_cf = simulate(cf_params, 30.0)
    describe(f"counterfactual (beta x {COUNTERFACTUAL_BETA_FACTOR})", cf_params, t_cf, Y_cf)

    fig_fit(days, in_bed, t, Y, params)
    fig_sir_curves(t, Y, params)
    fig_phase_portrait(params)
    fig_counterfactual(days, in_bed, t, Y, t_cf, Y_cf, params, cf_params)
    print("wrote figures/")

def fig_fit(days, in_bed, t, Y, params):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, Y[:, 1], color=BLUE, lw=2, label=f"fitted SIR, $R_0$ = {r0(params):.2f}")
    ax.scatter(days, in_bed, color=INK, zorder=3, label="boys in bed (BMJ 1978)")
    ax.set_xlabel("day")
    ax.set_ylabel("infected")
    ax.set_title("SIR model fitted to the 1978 boarding-school outbreak")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("figures/fit.png", dpi=150)
    plt.close(fig)

def fig_sir_curves(t, Y, params):
    fig, ax = plt.subplots(figsize=(7, 4))
    for col, color, label in ((0, GREEN, "S susceptible"), (1, RED, "I infected"), (2, BLUE, "R recovered")):
        ax.plot(t, Y[:, col], color=color, lw=2, label=label)
    peak = int(np.argmax(Y[:, 1]))
    ax.axvline(t[peak], color=GREY, ls="--", lw=1)
    ax.axhline(N / r0(params), color=GREY, ls=":", lw=1)
    ax.text(t[peak] + 0.4, N * 0.93, f"peak on day {t[peak]:.1f}", fontsize=9, color=INK)
    ax.text(21, N / r0(params) + 15, f"S = N / $R_0$ = {N / r0(params):.0f}", fontsize=9, color=INK)
    ax.set_xlabel("day")
    ax.set_ylabel("boys")
    ax.set_title("The three compartments over time")
    ax.legend(loc="center right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("figures/sir_curves.png", dpi=150)
    plt.close(fig)

def fig_phase_portrait(fitted):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    cases = [(1.5, GREY), (2.0, GREEN), (3.0, AMBER), (r0(fitted), BLUE)]
    for rho, color in cases:
        params = SIRParams(beta=rho * fitted.gamma, gamma=fitted.gamma, N=N)
        t, Y = simulate(params, 60.0)
        ax.plot(Y[:, 0], Y[:, 1], color=color, lw=2, label=f"$R_0$ = {rho:.2f}")
        peak = int(np.argmax(Y[:, 1]))
        ax.plot([N / rho, N / rho], [0, Y[peak, 1]], color=color, ls=":", lw=1)
        ax.scatter([Y[peak, 0]], [Y[peak, 1]], color=color, zorder=3, s=25)
    ax.annotate("start: S = 762, I = 1", xy=(N - I0, I0), xytext=(N - 20, 140), fontsize=9, ha="left",
                arrowprops=dict(arrowstyle="->", color=INK))
    ax.set_xlabel("S susceptible")
    ax.set_ylabel("I infected")
    ax.set_title("Phase portrait: the outbreak as a path in the (S, I) plane\n"
                 "each dot is a peak, and it sits exactly on the dotted line S = N / $R_0$", fontsize=10)
    ax.set_xlim(0, N + 10)
    ax.set_ylim(0, 360)
    ax.invert_xaxis()
    ax.legend(loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("figures/phase_portrait.png", dpi=150)
    plt.close(fig)

def fig_counterfactual(days, in_bed, t, Y, t_cf, Y_cf, params, cf_params):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(days, in_bed, color=INK, zorder=3, s=18, label="observed")
    ax.plot(t, Y[:, 1], color=BLUE, lw=2, label=f"fitted, $R_0$ = {r0(params):.2f}")
    ax.plot(t_cf, Y_cf[:, 1], color=AMBER, lw=2, ls="--",
            label=f"contacts down 20 %, $R_0$ = {r0(cf_params):.2f}")
    for tt, YY, color in ((t, Y, BLUE), (t_cf, Y_cf, AMBER)):
        peak = int(np.argmax(YY[:, 1]))
        ax.annotate(f"peak {YY[peak, 1]:.0f}", xy=(tt[peak], YY[peak, 1]), xytext=(tt[peak] + 3, YY[peak, 1] + 8),
                    fontsize=9, color=color, arrowprops=dict(arrowstyle="->", color=color))
    ax.set_xlabel("day")
    ax.set_ylabel("infected")
    ax.set_title("What if each boy had 20 % fewer infectious contacts?")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("figures/counterfactual.png", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    main()