import numpy as np
from pathlib import Path
import os
import shutil
import time
from matplotlib import pyplot as plt
import matplotlib.patches as patches

base_path = os.path.dirname(__file__)

from dyn.quad_10d_of_perception import Quad10d_OF
m = Quad10d_OF()
m.dt = 0.15

# x0 = np.array([0.9, 3.1, 1., 0.001, 0.001, 0.001, 0.001, -0.001, -0.001, -0.001]) 
# xG = np.array([-1.5, 2.5, 1.4, 0.001, 0.001, 0.001, -0.001, -0.001, -0.001, -0.001]) 

delay = True
N = 35 
data = np.load("data/quad10d_perception_0.npz")
primal_x = data["nominal_traj"]
primal_u = data["nominal_input"]
Phi_xx = data["Phi_xx"]
Phi_ux = data["Phi_ux"]
Phi_xy = data["Phi_xy"]
Phi_uy = data["Phi_uy"]
tube = data["backoff"]
tube_f = data["backoff_f"]
tubes = np.vstack([tube[:,:m.nx], tube_f[:m.nx]])

rollouts = np.load("data/rollout_0.npz")
rollout_states = rollouts["rollout_states"]
rollout_states_open = rollouts["rollout_states_open"]

fig, axes = plt.subplots(4, 3, figsize=(20,16))

for rollout in range(rollout_states.shape[0]):
    states = rollout_states[rollout, ...]
    states_open = rollout_states_open[rollout, ...]

    for idx in range(m.nx):
        row = int(idx / 3)
        col = idx % 3
        # add a label only for the first rollout so the legend has one entry
        axes[row][col].plot(
            np.arange(N + 1), states[idx, :],
            c='r', linestyle='--', linewidth=1.0, alpha=0.9,
            label='closed-loop rollout' if rollout == 0 else None
        )
        
        # axes[row][col].plot(
        #     np.arange(N + 1), states_open[idx, :],
        #     c='b', linestyle='--', linewidth=1.0, alpha=0.9,
        #     label='open-loop rollout' if rollout == 0 else None
        # )

# --- GREEN TUBES WITH ALPHA ---
state_names = [r'Position $p_x$ [m]', r'Position $p_y$ [m]', r'Heading $\theta$ [rad]', r'Speed $v$ [m/s]']
t = np.arange(N+1)

for idx in range(m.nx):
    row = int(idx/3)
    col = idx % 3

    # tube vector over the whole horizon (append terminal tube)
    tube_vec = np.concatenate([tube[:, idx], [tube_f[idx]]])

    upper = primal_x[idx, :] + tube_vec
    lower = primal_x[idx, :] - tube_vec

    # one shaded tube per subplot; label only on the first subplot
    axes[row][col].fill_between(
        t, lower, upper,
        color='tab:green', alpha=0.25, edgecolor='none',
        label='credal set state (closed-loop)'
    )

    # nominal
    axes[row][col].plot(t, primal_x[idx, :], label='optimal trajectory', color='green', linewidth=1.2)

    # axis labels
    axes[row][col].grid(True, linestyle='--', linewidth=0.6, alpha=0.4)
    # axes[row][col].set_ylabel(state_names[idx])
    if row  == 1:
        axes[row][col].set_xlabel('Time step $k$')

axes[0, 0].legend(loc='best')

# fig.savefig("imgs/rollout.pdf", format="pdf", dpi=300)
fig.savefig("imgs/rollout.png", format="png", dpi=300)