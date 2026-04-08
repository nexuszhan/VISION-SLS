from matplotlib import pyplot as plt
import matplotlib.patches as patch
from matplotlib.patches import Patch, Circle
from matplotlib.lines import Line2D
import numpy as np
import os
from PIL import Image, ImageDraw
from matplotlib.gridspec import GridSpec

from dyn.unicycle_of_perception import Unicycle_OF

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "figure.dpi": 300,
    "axes.labelsize": 20,
    "font.size": 15,  # larger base font
    "axes.titlesize": 18,
    "legend.fontsize": 20
})

fig = plt.figure(figsize=(34, 10), constrained_layout=True)
gs = GridSpec(
              nrows=4, ncols=4, figure=fig, 
              width_ratios=[1.2, 1., 1., 1.], 
              height_ratios=[1., 1., 1., 1.]
              )

ax_l = fig.add_subplot(gs[:, 0])  
ax_r  = fig.add_subplot(gs[:, 1])
axes = [[fig.add_subplot(gs[0,2]), fig.add_subplot(gs[0,3])],
        [fig.add_subplot(gs[1,2]), fig.add_subplot(gs[1,3])],
        [fig.add_subplot(gs[2,2]), fig.add_subplot(gs[2,3])],
        [fig.add_subplot(gs[3,2]), fig.add_subplot(gs[3,3])]
        ]   

obstacles = [(np.array([-3.55, 0.7]), 0.4), (np.array([-0.6, 1.15]), 0.35), (np.array([-1.2, -1.2]), 0.4)]
xmin = -3.5
xmax = 0.75
ymin = -2.0
ymax = 2.0
m = Unicycle_OF()
base_path = os.path.dirname(__file__)

scp_solution = np.load(os.path.join(base_path, "data/unicycle_of_perception_0.npz"))
X_fast = scp_solution['nominal_traj']
X_init = scp_solution['initial_traj']
backoff = scp_solution["backoff_x"]
initial_backoff = scp_solution["initial_backoff_x"]

scp_solution_no_info_gather = np.load(os.path.join(base_path, "data/unicycle_of_perception_CE_0.npz"))
X_fast_no_info_gather = scp_solution_no_info_gather['nominal_traj']
backoff_no_info_gather = scp_solution_no_info_gather["backoff_x"]

scp_solution_unrobust= np.load(os.path.join(base_path, "data/unicycle_of_perception_nonrobust_0.npz"))
X_fast_unrobust = scp_solution_unrobust['nominal_traj']
backoff_unrobust = scp_solution_unrobust["backoff_x"]

rollout_data = np.load(os.path.join(base_path, "data/rollout_0.npz"))
rollout_states = rollout_data["rollout_states"]
rollout_states_open = rollout_data["rollout_states_open"]

rollout_data_unrobust = np.load(os.path.join(base_path, "data/rollout_nonrobust_0.npz"))
rollout_states_unrobust = rollout_data_unrobust["rollout_states"]

ax = ax_l

def world_to_pixel(x, y, img_w, img_h, xmin, xmax, ymin, ymax):
    u = (x - xmin) / (xmax - xmin) * img_w
    v = (1 - (y - ymin) / (ymax - ymin)) * img_h  # image origin is top-left
    return int(round(u)), int(round(v))

def meters_to_pixels_radius(r_m, img_w, img_h, xmin, xmax, ymin, ymax):
    sx = img_w / float(xmax - xmin)
    sy = img_h / float(ymax - ymin)
    return int(round(r_m * min(sx, sy)))

file = "video_frames/frame_000.png"
ax.imshow(np.asarray(Image.open(file)))

im_w, im_h = Image.open(file).size
circles_px = []

for (center, radius_m) in obstacles:   # obstacles is [(np.array([x,y]), r), ...]
    u, v = world_to_pixel(center[0], center[1], im_w, im_h, xmin, xmax, ymin, ymax)
    r_px = meters_to_pixels_radius(radius_m, im_w, im_h, xmin, xmax, ymin, ymax)
    circles_px.append((u, v, r_px))

for (u, v, r) in circles_px:
    circ = Circle((u, v), radius=r, 
                          facecolor=(1, 0, 0, 0.3), 
                          edgecolor=(1, 0, 0, 0.3), lw=2)
    ax.add_patch(circ)

# traj_px = []
traj_u = []
traj_v = []
for t in range(X_fast.shape[1]):
    u, v = world_to_pixel(X_fast[0,t], X_fast[1,t], im_w, im_h, xmin, xmax, ymin, ymax)
    traj_u.append(u)
    traj_v.append(v)

ax.plot(traj_u, traj_v, color='g', lw=1, alpha=0.9, zorder=10)

tube_handles_nominal = []
tube_color_nominal = "tab:green"
tube_color_initial = "tab:red"
for i in range(X_fast.shape[1]):
    # nominal tube rectangle
    backoff_fast_sls = backoff[i, :]
    u, v = world_to_pixel(X_fast[0,i], X_fast[1,i], im_w, im_h, xmin, xmax, ymin, ymax)
    w = meters_to_pixels_radius(backoff_fast_sls[0], im_w, im_h, xmin, xmax, ymin, ymax)
    h = meters_to_pixels_radius(backoff_fast_sls[1], im_w, im_h, xmin, xmax, ymin, ymax)
    ret = m.plot_tube_as_rectangle(np.array([w,h]), np.array([u,v]), ax=ax)
    # get handle and recolor
    try:
        h = ret if isinstance(ret, patch.Patch) else ax.patches[-1]
    except Exception:
        h = ax.patches[-1]
    h.set_facecolor(tube_color_nominal)
    h.set_edgecolor(tube_color_nominal)
    h.set_alpha(0.5)

# ax.text(
#     0.02, 0.1, r"\textbf{(a)}",
#     transform=ax.transAxes,   # axes-relative coordinates
#     fontsize=40,
#     fontweight="bold",
#     color="white",            # or black, depending on background
#     va="top",
#     ha="left"
# )

ax.set_xlim(0, im_w)
ax.axis("off")
# ----------------------------------------------------------------------

ax = ax_r
# Disturbance map
cf = m.plot_disturbance_map(ax)
cbar = fig.colorbar(cf, ax=ax, shrink=0.95, aspect=30, pad=0.02)
cbar.set_label(r"Uncertainty $F^r(x)$", rotation=270, labelpad=16, fontsize=18)

xmin = -2.7
xmax = 0.5
ymin = -2.0
ymax = 2.0
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
# Obstacles
obstacle_handles = []
for center, radius in obstacles:
    circ = patch.Circle(center, radius, color="tab:red", alpha=0.8)
    ax.add_patch(circ)
    obstacle_handles.append(circ)

# optimal trajectory
# Tubes (different colors)
tube_color_nominal = "tab:green"

tube_handles_nominal = []
tube_handles_initial = []

for i in range(X_fast.shape[1]):
    # nominal tube rectangle
    backoff_fast_sls = backoff[i, :]
    ret = m.plot_tube_as_rectangle(backoff_fast_sls, X_fast[:, i], ax=ax)
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
    rollout_line, = ax.plot(rollout_states[i, 0, :], rollout_states[i, 1, :], 'g--', linewidth=1, label='Rollout')

# baseline: no information gathering
tube_color_no_info_gather = "tab:blue"  # pick a distinct color
tube_handles_no_info_gather = []

for i in range(X_fast_no_info_gather.shape[1]):
    backoff = backoff_no_info_gather[i, :]
    ret = m.plot_tube_as_rectangle(backoff, X_fast_no_info_gather[:, i], ax=ax)
    # get handle and style it
    h = ret if isinstance(ret, patch.Patch) else ax.patches[-1]
    h.set_facecolor(tube_color_no_info_gather)
    h.set_edgecolor(tube_color_no_info_gather)
    h.set_alpha(0.5)
    h.set_zorder(1)  # behind trajectories so lines stay visible
    tube_handles_no_info_gather.append(h)

no_info_gather_line, = ax.plot(X_fast_no_info_gather[0, :], X_fast_no_info_gather[1, :], 'b-', linewidth=1, label='CE trajectory')

# baseline: unrobust 
tube_color_unrobust = "tab:yellow"  # pick a distinct color
tube_handles_unrobust = []

unrobust_line, = ax.plot(X_fast_unrobust[0, :], X_fast_unrobust[1, :], 'y-', linewidth=1, label='unrobust trajectory')
for i in range(rollout_states_unrobust.shape[0]):
    unrobust_rollout_line, = ax.plot(rollout_states_unrobust[i, 0, :], rollout_states_unrobust[i, 1, :], 'y--', linewidth=1, label='Rollout')
# unrobust_rollout_line, = ax.plot(rollout_states_unrobust[8, 0, :], rollout_states_unrobust[8, 1, :], 'y--', linewidth=1, label='Rollout')
# ax.text(
#     0.01, 0.08, r"\textbf{(b)}",
#     transform=ax.transAxes,   # axes-relative coordinates
#     fontsize=40,
#     fontweight="bold",
#     color="black",            # or black, depending on background
#     va="top",
#     ha="left"
# )

# Build a clean legend (with proxies to avoid clutter)
# Legend handles (now real Line2D objects)
handles = [
    nominal_line,
    rollout_line,
    Patch(facecolor=tube_color_nominal, edgecolor=tube_color_nominal, alpha=0.5),
    no_info_gather_line,
    Patch(facecolor=tube_color_no_info_gather, edgecolor=tube_color_no_info_gather, alpha=0.5),
    unrobust_line,
    unrobust_rollout_line,
    # Patch(facecolor=tube_color_K0, edgecolor=tube_color_K0, alpha=0.5),
    Circle((0, 0), 0.2, color="tab:red", alpha=0.8)
]
labels = ["Optimal trajectory",
          "Optimal rollouts",
          "Optimal tubes\n(cost=2710)",
          "CE trajectory",
          "CE tubes\n(cost=3410)",
          "Non-robust trajectory",
          "Non-robust rollouts",
          "Obstacle"
]

# Put the legend INSIDE the axes (coords in axes units), so it never exceeds the plot box
leg = ax.legend(
    handles, labels,
    loc="lower right",  # corner of the legend box
    # bbox_to_anchor=(0.98, 0.98),  # place inside the axes
    # bbox_transform=ax.transAxes,  # interpret anchor in axes coords
    frameon=True, fancybox=True, framealpha=0.9,
    ncol=1,
    handlelength=1.6, handletextpad=0.6, labelspacing=0.2, borderaxespad=0.4
)
leg.get_frame().set_linewidth(0.8)

ax.set_xlabel("Position $p_x$ [m]")
ax.set_ylabel("Position $p_y$ [m]")
ax.grid(True, alpha=0.4)
ax.set_aspect('equal', adjustable='box')

# ---------------------------------
primal_x = scp_solution["nominal_traj"]
primal_u = scp_solution["nominal_input"]
tube = scp_solution["backoff"]
tube_f = scp_solution["backoff_f"]
N = 30

for rollout in range(rollout_states.shape[0]):
    states = rollout_states[rollout, ...]
    states_open = rollout_states_open[rollout, ...]
    for idx in range(m.nx):
        row = int(idx / 2)
        col = idx % 2 
        # add a label only for the first rollout so the legend has one entry
        axes[row][col].plot(
            np.arange(N + 1), states[idx, :],
            c='r', linestyle='--', linewidth=1.0, alpha=0.9,
            label='closed-loop rollout' if rollout == 0 else None
        )
        
        axes[row][col].plot(
            np.arange(N + 1), states_open[idx, :],
            c='b', linestyle='--', linewidth=1.0, alpha=0.9,
            label='open-loop rollout' if rollout == 0 else None
        )

# --- GREEN TUBES WITH ALPHA ---
state_names = [r'Position $p_x$ [m]', r'Position $p_y$ [m]', r'Heading $\theta$ [rad]', r'Speed $v$ [m/s]']
t = np.arange(N+1)

for idx in range(m.nx):
    row = int(idx/2)
    col = idx % 2 

    # tube vector over the whole horizon (append terminal tube)
    tube_vec = np.concatenate([tube[:, idx], [tube_f[idx]]])

    upper = primal_x[idx, :] + tube_vec
    lower = primal_x[idx, :] - tube_vec

    # one shaded tube per subplot; label only on the first subplot
    axes[row][col].fill_between(
        t, lower, upper,
        color='tab:green', alpha=0.25, edgecolor='none',
        label='reachable set'
    )

    # nominal
    axes[row][col].plot(t, primal_x[idx, :], label='optimal trajectory', color='green', linewidth=1.2)

    # axis labels
    axes[row][col].grid(True, linestyle='--', linewidth=0.6, alpha=0.4)
    axes[row][col].set_ylabel(state_names[idx])
    # if row  == 1:
    #     axes[row][col].set_xlabel('Time step $k$')

axes[0][0].legend(loc='best', labelspacing=0.2)

axes[0][1].text(
    0.9, 0.95, r"\textbf{(c)}",
    transform=axes[0][1].transAxes,   # axes-relative coordinates
    fontsize=40,
    fontweight="bold",
    color="black",            # or black, depending on background
    va="top",
    ha="left"
)

# ----------------------
tube = scp_solution["backoff_x"]
tube_no_info = scp_solution_no_info_gather["backoff_x"]

axes[2][0].plot(np.arange(N+1), tube[:,0], color=tube_color_nominal, label="optimal")
axes[2][0].plot(np.arange(N+1), tube_no_info[:,0], color=tube_color_no_info_gather, label="CE")
# axes[2][0].set_title("px")
axes[2][0].set_ylabel("Tube width of $p_x$ [m]")
axes[2][0].legend(labelspacing=0.2)

axes[2][1].plot(np.arange(N+1), tube[:,1], color=tube_color_nominal)
axes[2][1].plot(np.arange(N+1), tube_no_info[:,1], color=tube_color_no_info_gather)
# axes[2][1].set_title("py")
axes[2][1].set_ylabel("Tube width of $p_y$ [m]")

axes[3][0].plot(np.arange(N+1), tube[:,2], color=tube_color_nominal)
axes[3][0].plot(np.arange(N+1), tube_no_info[:,2], color=tube_color_no_info_gather)
# axes[3][0].set_title("v")
axes[3][0].set_xlabel("Time step k")
axes[3][0].set_ylabel(r"Tube width of $\theta$ [rad]")

axes[3][1].plot(np.arange(N+1), tube[:,3], color=tube_color_nominal)
axes[3][1].plot(np.arange(N+1), tube_no_info[:,3], color=tube_color_no_info_gather)
# axes[3][1].set_title("theta")
axes[3][1].set_xlabel("Time step k")
axes[3][1].set_ylabel("Tube width of $v$ [m/s]")

axes[3][1].text(
    0.89, 0.3, r"\textbf{(d)}",
    transform=axes[3][1].transAxes,   # axes-relative coordinates
    fontsize=40,
    fontweight="bold",
    color="black",            # or black, depending on background
    va="top",
    ha="left"
)

axes[2][0].grid(True, linestyle='--', linewidth=0.6, alpha=0.4)
axes[2][1].grid(True, linestyle='--', linewidth=0.6, alpha=0.4)
axes[3][0].grid(True, linestyle='--', linewidth=0.6, alpha=0.4)
axes[3][1].grid(True, linestyle='--', linewidth=0.6, alpha=0.4)

# file_path = "imgs/parking_lot_full.pdf"
# plt.savefig(file_path, format="pdf", bbox_inches='tight')
file_path = "imgs/parking_lot_full.png"
plt.savefig(file_path, format="png", bbox_inches='tight')

# leg.remove()
# ax_r.set_xlim(-0.75, -0.25)
# ax_r.set_ylim(0.75, 1.25)
# zoom_file_path = "imgs/parking_lot_zoom.png"
# plt.savefig(zoom_file_path, format="png", dpi=300, bbox_inches='tight')
# plt.close()