from matplotlib import pyplot as plt
import matplotlib.patches as patch
from matplotlib.patches import Patch, Circle, Rectangle
from matplotlib.lines import Line2D
import numpy as np
import os
from PIL import Image, ImageDraw

from dyn.light_dark_of import LightDark_OF

m = LightDark_OF()
base_path = os.path.dirname(__file__)

scp_solution = np.load(os.path.join(base_path, "data/light-dark.npz"))
X_fast = scp_solution['nominal_traj']
X_fast_no_info_gather = scp_solution['initial_traj']
backoff = scp_solution["backoff_x"]
backoff_no_info_gather = scp_solution["initial_backoff_x"]

rollout_data = np.load(os.path.join(base_path, "data/rollout.npz"))
rollout_states = rollout_data["rollout_states"]
feedback_error = rollout_data["feedback_error"]
feedback_error_terminal = rollout_data["feedback_error_terminal"]
openloop_error = rollout_data["openloop_error"]
openloop_error_terminal = rollout_data["openloop_error_terminal"]
# print("feedback error: {:.3f}, {:.3f}".format(feedback_error, feedback_error_terminal))
# print("openloop error: {:.3f}, {:.3f}".format(openloop_error, openloop_error_terminal))

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "figure.dpi": 300,
    "axes.labelsize": 13
})

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# ----------------------------------------------------------------------

# Disturbance map
cf = m.plot_disturbance_map(ax)
cbar = fig.colorbar(cf, ax=ax, shrink=0.35, aspect=30, pad=0.02)
cbar.set_label(r"Uncertainty $e_\beta(x)$", rotation=270, labelpad=16, fontsize=14)

xmin = -2.
xmax = 4.
ymin = -0.5
ymax = 2.5
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# baseline: no information gathering
tube_color_no_info_gather = "tab:blue"  # pick a distinct color
tube_handles_no_info_gather = []

for i in range(X_fast_no_info_gather.shape[1]):
    ret = m.plot_tube_as_rectangle(backoff_no_info_gather[i, :], X_fast_no_info_gather[:, i], ax=ax)
    # get handle and style it
    h = ret if isinstance(ret, patch.Patch) else ax.patches[-1]
    h.set_facecolor(tube_color_no_info_gather)
    h.set_edgecolor(tube_color_no_info_gather)
    h.set_alpha(0.5)
    h.set_zorder(1)  # behind trajectories so lines stay visible
    tube_handles_no_info_gather.append(h)

no_info_gather_line, = ax.plot(X_fast_no_info_gather[0, :], X_fast_no_info_gather[1, :], 'b-', linewidth=1, label='CE trajectory')

# optimal trajectory
# Tubes (different colors)
tube_color_nominal = "tab:green"

tube_handles_nominal = []
tube_handles_initial = []

for i in range(X_fast.shape[1]):
    # nominal tube rectangle
    ret = m.plot_tube_as_rectangle(backoff[i, :], X_fast[:, i], ax=ax)
    # get handle and recolor
    try:
        h = ret if isinstance(ret, patch.Patch) else ax.patches[-1]
    except Exception:
        h = ax.patches[-1]
    h.set_facecolor(tube_color_nominal)
    h.set_edgecolor(tube_color_nominal)
    h.set_alpha(0.5)
    tube_handles_nominal.append(h)

nominal_line, = ax.plot(X_fast[0, :], X_fast[1, :], 'g-', linewidth=1, label='Optimal trajectory')
for i in range(rollout_states.shape[0]):
    rollout_line, = ax.plot(rollout_states[i, 0, :], rollout_states[i, 1, :], 'g--', linewidth=0.25, label='Rollout')

x_min_f = m.x_f_min
x_max_f = m.x_f_max
ax.add_patch(Rectangle(
    (x_min_f[0], x_min_f[1]),
    x_max_f[0] - x_min_f[0],
    x_max_f[1] - x_min_f[1],
    fill=False,
    edgecolor='k',
    linewidth=1.,
    label='Terminal constraint'
))

# Build a clean legend (with proxies to avoid clutter)
# Legend handles (now real Line2D objects)
handles = [
    nominal_line,
    rollout_line,
    no_info_gather_line,
    Patch(facecolor=tube_color_nominal, edgecolor=tube_color_nominal, alpha=0.5),
    Patch(facecolor=tube_color_no_info_gather, edgecolor=tube_color_no_info_gather, alpha=0.5),
    Patch(fill=False, edgecolor='k'),
]
labels = ["Optimal reference",
          "Closed-loop rollouts",
          "CE reference",
          "Optimal tube (Cost = 660)",
          "CE: post-hoc tube (Cost = 832)",
          "Terminal constraint",
]

# Put the legend INSIDE the axes (coords in axes units), so it never exceeds the plot box
leg = ax.legend(
    handles, labels,
    loc="upper right",  # corner of the legend box
    # bbox_to_anchor=(0.98, 0.98),  # place inside the axes
    # bbox_transform=ax.transAxes,  # interpret anchor in axes coords
    frameon=True, fancybox=True, framealpha=0.9,
    ncol=1,
    handlelength=1.6, handletextpad=0.6, labelspacing=0.2, borderaxespad=0.2
)
leg.get_frame().set_linewidth(0.8)

ax.set_xlabel("Position $p_x$ [m]")
ax.set_ylabel("Position $p_y$ [m]")
ax.grid(True, alpha=0.4)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()

# file_path = "imgs/light_dark.pdf"
# plt.savefig(file_path, format="pdf", dpi=300, bbox_inches='tight')
file_path = "imgs/light_dark.png"
plt.savefig(file_path, format="png", dpi=300, bbox_inches='tight')
