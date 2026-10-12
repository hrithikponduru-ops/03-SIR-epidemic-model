"""Conceptual diagrams for README.md (data-independent). Writes to figures/.

Run from this folder: python make_figures.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sir.rk4 import euler_step, integrate, rk4_step

INK, BLUE, GREEN, RED, GREY, AMBER = "#1f2937", "#2563eb", "#059669", "#dc2626", "#9ca3af", "#d97706"

def fig_convergence_order():
    decay = lambda t, y, p: -y
    hs = np.array([0.4, 0.2, 0.1, 0.05, 0.025, 0.0125])
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for step, color, label, order in ((euler_step, RED, "Euler", 1), (rk4_step, BLUE, "RK4", 4)):
        errors = [abs(integrate(decay, np.array([1.0]), 0.0, 2.0, h, None, step=step)[1][-1, 0] - np.exp(-2.0))
                  for h in hs]
        ax.loglog(hs, errors, "o-", color=color, lw=2, label=f"{label}: measured error")
        ref = errors[0] * (hs / hs[0]) ** order
        ax.loglog(hs, ref, ls="--", color=color, lw=1, alpha=0.7, label=f"slope {order} reference line")
    ax.set_xlabel("step size h")
    ax.set_ylabel("error in y(2) for y' = -y")
    ax.set_title("Halve h: Euler's error halves, RK4's drops by 16")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("figures/convergence_order.png", dpi=150)
    plt.close(fig)

def fig_rk4_slopes():
    """One RK4 step on y' = y from y(0) = 1 with h = 1, where the slopes differ visibly."""
    f = lambda t, y: y
    y0, h = 1.0, 1.0
    k1 = f(0, y0)
    k2 = f(h / 2, y0 + h / 2 * k1)
    k3 = f(h / 2, y0 + h / 2 * k2)
    k4 = f(h, y0 + h * k3)
    y1 = y0 + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    tt = np.linspace(0, 1, 100)
    ax.plot(tt, np.exp(tt), color=INK, lw=2, label="exact solution $y = e^t$")
    seg = 0.18
    for (t_pt, y_pt, k, color, name) in (
        (0.0, y0, k1, BLUE, "$k_1$ = slope at start = 1"),
        (0.5, y0 + h / 2 * k1, k2, GREEN, "$k_2$ = slope at midpoint guess = 1.5"),
        (0.5, y0 + h / 2 * k2, k3, AMBER, "$k_3$ = slope at better midpoint = 1.75"),
        (1.0, y0 + h * k3, k4, RED, "$k_4$ = slope at end guess = 2.75"),
    ):
        ax.plot([t_pt - seg, t_pt + seg], [y_pt - seg * k, y_pt + seg * k], color=color, lw=2.5)
        ax.scatter([t_pt], [y_pt], color=color, zorder=3, s=30, label=name)
    ax.scatter([1.0], [y0 + h * k1], color=GREY, marker="s", zorder=3, s=40, label=f"Euler: $y_1$ = {y0 + h * k1:.3f}")
    ax.scatter([1.0], [y1], color=INK, marker="*", zorder=4, s=140, label=f"RK4: $y_1$ = {y1:.4f}")
    ax.annotate(f"exact: e = {np.e:.4f}", xy=(1.0, np.e), xytext=(1.06, 2.45), fontsize=9,
                arrowprops=dict(arrowstyle="->", color=INK))
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.set_title("One RK4 step: four slope samples, weighted 1, 2, 2, 1")
    ax.set_xlim(-0.25, 1.4)
    ax.set_ylim(0.6, 3.4)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("figures/rk4_slopes.png", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    fig_convergence_order()
    fig_rk4_slopes()
    print("wrote figures/")