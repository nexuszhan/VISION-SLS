from util.plot_3d import *
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image, ImageDraw

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "figure.dpi": 300,
    "axes.labelsize": 20,
    "font.size": 15,  # larger base font
    "axes.titlesize": 18,
    "legend.fontsize": 20
})


name = "Quad10d_OF_Perception"
obstacles = [(np.array([1., 1.]), 0.2), (np.array([-1.5, 4.]), 0.2), (np.array([0.5, 3.]), 0.2),
                 (np.array([-0.5, 1.]), 0.2), (np.array([-1., 2.5]), 0.2)]
base_path = os.path.dirname(__file__)

scp_solution = np.load(os.path.join(base_path, "data/quad10d_perception_0.npz"))
X_fast = scp_solution['nominal_traj']
backoff_fast = scp_solution['backoff_x']  

scp_solution_no_info_gather = np.load(os.path.join(base_path, "data/quad10d_perception_CE_0.npz"))
X_fast_no_info_gather = scp_solution_no_info_gather['nominal_traj']
backoff_fast_no_info_gather = scp_solution_no_info_gather['backoff_x']  

scp_solution_unrobust = np.load(os.path.join(base_path, "data/quad10d_perception_nonrobust_0.npz"))
X_fast_unrobust = scp_solution_unrobust['nominal_traj']
backoff_fast_unrobust = scp_solution_unrobust['backoff_x']

rollout_data = np.load(os.path.join(base_path, "data/rollout_0.npz"))
rollout_states = rollout_data["rollout_states"]
feedback_error = rollout_data["feedback_error"]
feedback_error_terminal = rollout_data["feedback_error_terminal"]
openloop_error = rollout_data["openloop_error"]
openloop_error_terminal = rollout_data["openloop_error_terminal"]
avg_inference_time = rollout_data["avg_inference_time"]

rollout_data_unrobust = np.load(os.path.join(base_path, "data/rollout_nonrobust_0.npz"))
rollout_states_unrobust = rollout_data_unrobust["rollout_states"]

print("feedback error: {:.3f}, {:.3f}".format(feedback_error, feedback_error_terminal))
print("openloop error: {:.3f}, {:.3f}".format(openloop_error, openloop_error_terminal))
print("average inference time: {:.3f}".format(avg_inference_time))

# positions
px, py, pz = X_fast[0, :], X_fast[1, :], X_fast[2, :]
px0, py0, pz0 = X_fast_no_info_gather[0, :], X_fast_no_info_gather[1, :], X_fast_no_info_gather[2, :]

# per-step half-sizes for boxes (x,y,z)
hf = np.abs(backoff_fast[:, :3])
hi = np.abs(backoff_fast_no_info_gather[:, :3])

# bounds including boxes
x_min = min((px - hf[:, 0]).min(), (px0 - hi[:, 0]).min())
x_max = max((px + hf[:, 0]).max(), (px0 + hi[:, 0]).max())
y_min = min((py - hf[:, 1]).min(), (py0 - hi[:, 1]).min())
y_max = max((py + hf[:, 1]).max(), (py0 + hi[:, 1]).max())
z_min = min((pz - hf[:, 2]).min(), (pz0 - hi[:, 2]).min())
z_max = max((pz + hf[:, 2]).max(), (pz0 + hi[:, 2]).max())


fig = plt.figure(figsize=(12, 6),layout='constrained')
gs = gridspec.GridSpec(2, 4, figure=fig, width_ratios=[4, 1, 1, 1])

ax = fig.add_subplot(gs[:, 0], projection='3d')

obstalce_proxy = plot_3d_obstacles(obstacles, z_min, z_max, ax)

line_fast, tube_fast, line_rollout = plot_3d_rollout(X_fast, hf, "tab:green", "Optimal", x_min, x_max, y_min, y_max, z_min, z_max, ax, rollout_states)
line_no_info_gather, tube_fast_no_info_gather, _ = plot_3d_rollout(X_fast_no_info_gather, hi, "tab:blue", "CE", x_min, x_max, y_min, y_max, z_min, z_max, ax)
line_fast_unrobust, tube_fast_unrobust, line_rollout_unrobust = plot_3d_rollout(X_fast_unrobust, np.abs(backoff_fast_unrobust[:, :3]), "tab:orange", "Non-robust", 
                                                            x_min, x_max, y_min, y_max, z_min, z_max, ax, rollout_states_unrobust)
ax.legend(handles=[line_fast, tube_fast, line_rollout, 
                   line_no_info_gather, tube_fast_no_info_gather, 
                   line_fast_unrobust, line_rollout_unrobust, 
                   obstalce_proxy], loc="upper left", fontsize=12, ncol=2)
# tick labels
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='z', which='major', labelsize=12)

ax.set_box_aspect([1, 1, 0.7])
ax.grid(True)

# enforce equal x–y scale so circles look like circles
xr = x_max - x_min
yr = y_max - y_min
r = 0.5 * max(xr, yr)
xmid = 0.5 * (x_max + x_min)
ymid = 0.5 * (y_max + y_min)

ax.set_xlim(xmid - r, xmid + r)
ax.set_ylim(ymid - r, ymid + r)
ax.set_box_aspect([1, 1, 0.7])  # keep x and y visually square

# ----------------------------------------------
axes = []
plot_steps = [34, 28, 24, 16, 10, 0]
for row in range(2):
    for col in range(1, 4):
        ax = fig.add_subplot(gs[row, col])
        t = plot_steps.pop()
        file = f"video_frames/frame_{t:03d}.png"
        ax.imshow(np.asarray(Image.open(file)))
        ax.set_title(f"step={t:d}")
        ax.axis("off")
        axes.append(ax)

fig.savefig("imgs/quad_perception.png", format="png", dpi=300, bbox_inches='tight')
# fig.savefig("imgs/quad_perception.pdf", format="pdf", dpi=300, bbox_inches='tight')

# ------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(10,6))
import matplotlib.patches as patch

for obstacle in obstacles:
    p = patch.Circle(obstacle[0], obstacle[1], fill=False, 
                    edgecolor="tab:red", linewidth=2, alpha=0.8)
    ax.add_patch(p)

for i in range(X_fast.shape[1]):
    # plot rectangle based on backoff
    backoff_fast_sls = backoff_fast[i, :]
    width, height = 2*backoff_fast_sls[0], 2*backoff_fast_sls[1]
    x_center, y_center = X_fast[0,i], X_fast[1,i]

    # Calculate bottom-left corner based on center
    bottom_left_x = x_center - width / 2
    bottom_left_y = y_center - height / 2

    # Create a rectangle with the specified attributes
    rectangle = patch.Rectangle(
        (bottom_left_x, bottom_left_y),  # Bottom-left corner
        width,  # Width
        height,  # Height
        facecolor='lightgreen',  # Light green background
        edgecolor='green',  # Green edges
        linewidth=2,  # Edge line width
        alpha=0.3  # Transparency
    )
    ax.add_patch(rectangle)
    # ----------------------
    backoff_initial = backoff_fast_no_info_gather[i,:]
    width, height = 2*backoff_initial[0], 2*backoff_initial[1]
    x_center, y_center = X_fast_no_info_gather[0,i], X_fast_no_info_gather[1,i]

    # Calculate bottom-left corner based on center
    bottom_left_x = x_center - width / 2
    bottom_left_y = y_center - height / 2

    # Create a rectangle with the specified attributes
    rectangle = patch.Rectangle(
        (bottom_left_x, bottom_left_y),  # Bottom-left corner
        width,  # Width
        height,  # Height
        facecolor='lightgreen',  # Light green background
        edgecolor='green',  # Green edges
        linewidth=2,  # Edge line width
        alpha=0.3  # Transparency
    )
    ax.add_patch(rectangle)

ax.plot(X_fast[0, :], X_fast[1, :], "tab:green", linewidth=1)
for i in range(rollout_states.shape[0]):
    ax.plot(rollout_states[i, 0, :], rollout_states[i, 1, :], "tab:green", linestyle='--', linewidth=1)

ax.plot(X_fast_no_info_gather[0, :], X_fast_no_info_gather[1, :], "tab:blue", linewidth=1)

ax.plot(X_fast_unrobust[0,:], X_fast_unrobust[1,:], "tab:orange", linewidth=1)
for i in range(rollout_states_unrobust.shape[0]):
    ax.plot(rollout_states_unrobust[i, 0, :], rollout_states_unrobust[i, 1, :], "tab:orange", linestyle='--', linewidth=1)

ax.set_xlim(-1.5, -0.5)
ax.set_ylim(2., 3.)
ax.set_aspect('equal', adjustable='box')

fig.savefig("imgs/quad_perception_zoom.png", format="png", dpi=300, bbox_inches="tight")
