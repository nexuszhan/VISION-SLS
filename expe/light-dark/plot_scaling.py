from matplotlib import pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "figure.dpi": 300,
    "axes.labelsize": 13
})

horizon = np.array([20, 30, 40, 50, 60], dtype=float)

ours_time = np.array([1.129, 4.325, 6.036, 7.219, 9.309], dtype=float)
optimal_time = np.array([51.966, 129.04, 247.346, 429.482, 726.721], dtype=float)
ours_tube_cost = np.array([659.612, 795.324, 930.345, 1065.365, 1200.385], dtype=float)
optimal_tube_cost = np.array([659.602, 795.317, 930.337, 1065.355, 1200.375], dtype=float)

log_ours_time = np.log10(ours_time)
log_optimal_time = np.log10(optimal_time)
norm_ours_tube_cost = (ours_tube_cost - np.min(optimal_tube_cost)) / np.max(optimal_tube_cost)
norm_optimal_tube_cost = (optimal_tube_cost - np.min(optimal_tube_cost)) / np.max(optimal_tube_cost)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes[0]
ax.plot(horizon, log_ours_time, color='g', label="Double Riccati (ours)")
ax.plot(horizon, log_optimal_time, color='b', label="MOSEK")

ax.text(
    0.02, 0.95, r"\textbf{(a)}",
    transform=ax.transAxes,   # axes-relative coordinates
    fontsize=20,
    fontweight="bold",
    color="black",            # or black, depending on background
    va="top",
    ha="left"
)

ax.set_xlabel("Horizon")
ax.set_ylabel("Log Runtime (s)")
ax.legend(fontsize=7)

ax = axes[1]
ax.plot(horizon, norm_ours_tube_cost, color='g', label="Double Riccati (ours)")
ax.plot(horizon, norm_optimal_tube_cost, color='b', label="MOSEK")

ax.text(
    0.02, 0.95, r"\textbf{(b)}",
    transform=ax.transAxes,   # axes-relative coordinates
    fontsize=20,
    fontweight="bold",
    color="black",            # or black, depending on background
    va="top",
    ha="left"
)

ax.set_xlabel("Horizon")
ax.set_ylabel("Normalized Tube Cost")
# ax.legend()

fig.savefig("imgs/scaling.png", format="png", dpi=300)
# fig.savefig("imgs/scaling.pdf", format="pdf", dpi=300)