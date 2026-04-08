import numpy as np
from pathlib import Path
import os
import shutil
import time
from matplotlib import pyplot as plt
import matplotlib.patches as patches

base_path = os.path.dirname(__file__)
device = 'cpu' 

num_row = 2
num_col = 2
fig, axes = plt.subplots(num_row, num_col,figsize=(10, 6), constrained_layout=True)
# num_row = 6
# num_col = 10
# fig, axes = plt.subplots(num_row, num_col,figsize=(100, 60), constrained_layout=True)

data = np.load(os.path.join(base_path, "data/G1_robust.npz")) 
primal_x = data["nominal_traj"]
primal_u = data["nominal_input"]
Phi_xx = data["Phi_xx"]
Phi_ux = data["Phi_ux"]
Phi_xy = data["Phi_xy"]
Phi_uy = data["Phi_uy"]
tube = data["backoff"]
tube_f = data["backoff_f"]

x0 = primal_x[:,0]
N = primal_x.shape[1] - 1 

rollouts = np.load("data/G1_rollout.npz")
rollout_states = rollouts["rollout_states"]
rollout_states_open = rollouts["rollout_states_open"]

# state_indices = [0, 2, 4, 6] # for G1_rollout
# state_indices = [i for i in range(59)] # for G1_rollout_all
state_indices = [20, 21, 22, 23]

# state_indices = [0, 1, 8, 9]
# state_indices = [2, 3, 10, 11]
# state_indices = [4, 5, 12, 13]
# state_indices = [6, 7, 14, 15]
# state_indices = [16, 17, 24, 25]
# state_indices = [18, 19, 26, 27]
# state_indices = [20, 21, 28, 29]
# state_indices = [22, 23, 30, 31]
# state_indices = [32, 33, 40, 41]
# state_indices = [34, 35, 42, 43]
# state_indices = [36, 37, 44, 45]
# state_indices = [38, 39, 46, 47]
# state_indices = [48, 49, 56, 57]
# state_indices = [50, 51, 58, 0]
# state_indices = [52, 53, 0, 0]
# state_indices = [54, 55, 0, 0]

state_names = [r'Position $p_x$ [m] ', r'Position $p_y$ [m]', r'Position $p_z$ [m]', 
               r'Quaternion $q_x$', r'Quaternion $q_y$', r'Quaternion $q_z$', r'Quaternion $q_w$',
               r'Left hip pitch', r'Left hip roll', r'Left hip yaw', r'Left knee', r'Left ankle pitch', r'Left ankle roll',
               r'Right hip pitch', r'Right hip roll', r'Right hip yaw', r'Right knee', r'Right ankle pitch', r'Right ankle roll',
               r'Waist yaw',
               r'Left shoulder pitch', r'Left shoulder roll', r'Left shoulder yaw', r'Left elbow', r'Left wirst',
               r'Right shoulder pitch', r'Right shoulder roll', r'Right shoulder yaw', r'Right elbow', r'Right wirst',
               r'Velocity $v_x$ [m/s]', r'Velocity $v_y$ [m/s]', r'Velocity $v_z$ [m/s]',
               r'Velocity $w_x$ [rad/s]', r'Velocity $w_y$ [rad/s]', r'Velocity $w_z$ [rad/s]',
               r'Left hip pitch vel', r'Left hip roll vel', r'Left hip yaw vel', r'Left knee vel', r'Left ankle pitch vel', r'Left ankle roll vel',
               r'Right hip pitch vel', r'Right hip roll vel', r'Right hip yaw vel', r'Right knee vel', r'Right ankle pitch vel', r'Right ankle roll vel',
               r'Waist yaw vel',
               r'Left shoulder pitch vel', r'Left shoulder roll vel', r'Left shoulder yaw vel', r'Left elbow vel', r'Left wirst vel',
               r'Right shoulder pitch vel', r'Right shoulder roll vel', r'Right shoulder yaw vel', r'Right elbow vel', r'Right wirst vel']
for i, state_idx in enumerate(state_indices):
    row = int(i/num_col) - 0
    col = i % num_col
    for t in range(N):
        axes[row][col].plot([t-0.1, t+0.1], [primal_x[state_idx,t]+tube[t,state_idx], primal_x[state_idx,t]+tube[t,state_idx]], 'k-', linewidth=1)
        axes[row][col].plot([t-0.1, t+0.1], [primal_x[state_idx,t]-tube[t,state_idx], primal_x[state_idx,t]-tube[t,state_idx]], 'k-', linewidth=1)
    axes[row][col].plot([N-0.1, N+0.1], [primal_x[state_idx,N]+tube_f[state_idx], primal_x[state_idx,N]+tube_f[state_idx]], 'k-', linewidth=1)
    axes[row][col].plot([N-0.1, N+0.1], [primal_x[state_idx,N]-tube_f[state_idx], primal_x[state_idx,N]-tube_f[state_idx]], 'k-', linewidth=1)

    axes[row][col].plot(np.arange(N+1), primal_x[state_idx,:])


for rollout in range(rollout_states.shape[0]):
    states = rollout_states[rollout, ...]
    states_open = rollout_states_open[rollout, ...]
    for i, state_idx in enumerate(state_indices):
        row = int(i / num_col)
        col = i % num_col
        # add a label only for the first rollout so the legend has one entry
        axes[row][col].plot(
            np.arange(N + 1), states[state_idx, :],
            c='r', linestyle='--', linewidth=1.0, alpha=0.9,
            label='closed-loop rollout' if rollout == 0 else None
        )
        # todo: udpdate the plots! Swap closed-loop and open-loop
        axes[row][col].plot(
            np.arange(N + 1), states_open[state_idx, :],
            c='b', linestyle='--', linewidth=1.0, alpha=0.9,
            label='open-loop rollout' if rollout == 0 else None
        )

# --- GREEN TUBES WITH ALPHA ---
t = np.arange(N+1)

for i, state_idx in enumerate(state_indices):
    row = int(i/num_col)
    col = i % num_col

    # tube vector over the whole horizon (append terminal tube)
    tube_vec = np.concatenate([tube[:, state_idx], [tube_f[state_idx]]])

    upper = primal_x[state_idx, :] + tube_vec
    lower = primal_x[state_idx, :] - tube_vec

    # one shaded tube per subplot; label only on the first subplot
    axes[row][col].fill_between(
        t, lower, upper,
        color='tab:green', alpha=0.25, edgecolor='none',
        label='credal set state (closed-loop)'
    )

    # nominal
    axes[row][col].plot(t, primal_x[state_idx, :], label='optimal trajectory', color='green', linewidth=1.2)

    # axis labels
    axes[row][col].grid(True, linestyle='--', linewidth=0.6, alpha=0.4)
    axes[row][col].set_ylabel(state_names[state_idx])

    if row == num_row:
        axes[row][col].set_xlabel('Time step $k$')

axes[0, 0].legend(loc='best')

fig.savefig("imgs/G1_rollout.png", format="png")
# fig.savefig("imgs/G1_rollout.pdf", format="pdf", dpi=300)
# fig.savefig("imgs/G1_rollout_all.pdf", format="pdf", dpi=300)
