from dyn.turtlebot_of import Turtlebot_OF
import numpy as np
from solver.SCP_OF_SLS import SCP_OF_SLS
from datetime import datetime
import os
import time
from matplotlib import pyplot as plt
from util.footnote import add_footnote_time
import matplotlib.gridspec as gridspec
from PIL import Image, ImageDraw

if __name__ == '__main__':
    m = Turtlebot_OF()

    N = 60 # horizon length

    m.dt = 0.2

    robot_radii = 0.2 
    obstacles = [(np.array([0.745, 0.656]), 0.432+robot_radii), 
                 (np.array([2.66, 0.72]), 0.254+robot_radii)
                 ]
    
    x0 = np.array([0., 0., 0., 0., 0.]) 
    xG = np.array([3., 1.5, 0., 0., 0.]) 
    
    scp_solution = np.load("Turtlebot_OF_DINO.npz")

    # Create a figure for plotting
    plt.rcParams.update({
        "font.size": 14,  # larger base font
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "legend.fontsize": 13
    })
    fig = plt.figure(figsize=(12, 6),layout='constrained')
    gs = gridspec.GridSpec(2, 4, figure=fig, width_ratios=[4, 1, 1, 1])
    
    ax = fig.add_subplot(gs[:, 0])

    X_fast = scp_solution['nominal_traj']

    import matplotlib.patches as patch
    for i, obstacle in enumerate(obstacles):
        if i == 0:
            p = patch.Circle(obstacle[0], obstacle[1], color="tab:red", alpha=0.8, label="Obstacle")
        else:
            p = patch.Circle(obstacle[0], obstacle[1], color="tab:red", alpha=0.8)
        ax.add_patch(p)

    for i in range(scp_solution['nominal_traj'].shape[1]):
        # plot rectangle based on backoff
        backoff_fast_sls = scp_solution['backoff_x'][i, :]
        m.plot_tube_as_rectangle(backoff_fast_sls, X_fast[:, i], ax=ax)
        h = ax.patches[-1]
        h.set_facecolor("tab:green")
        h.set_edgecolor("tab:green")
        h.set_alpha(0.3)

    # Plot nominal trajectory
    ax.plot(X_fast[0,:], X_fast[1,:], 'g-', linewidth=2.5, label="nominal")

    files = ["visited_poses_DINO_1_6.npz", "visited_poses_DINO_1_10.npz", "visited_poses_DINO_1_11.npz"]

    for i, file in enumerate(files):
        # label = f"rollout {i+1}"
        actual_traj = np.load(file)
        actual_traj = actual_traj["visited_poses"]
        # ax.plot(actual_traj[0,:], actual_traj[1,:], 'b--', linewidth=2.5, label=label)
        if i == 0:
            ax.plot(actual_traj[0,:], actual_traj[1,:], 'b--', linewidth=2.5, label="Rollout")
        else:
            ax.plot(actual_traj[0,:], actual_traj[1,:], 'b--', linewidth=2.5)

        for i in range(scp_solution['nominal_traj'].shape[1]):
            backoff_fast_sls = scp_solution['backoff_x'][i, :]
            if abs(X_fast[0,i] - actual_traj[0,i]) > backoff_fast_sls[0] or \
                abs(X_fast[1,i] - actual_traj[1,i]) > backoff_fast_sls[1]:
                print("out of tube: ", i)
                print(abs(X_fast[0,i] - actual_traj[0,i]), " ", backoff_fast_sls[0])
                print(abs(X_fast[1,i] - actual_traj[1,i]), " ", backoff_fast_sls[1])

    ax.legend()

    # ax.set_title("SCP-OF-SLS")
    ax.set_xlabel("$p_x$")
    ax.set_ylabel("$p_y$")
    ax.grid(True)
    ax.set_aspect('equal', adjustable='datalim')

    # ----------------------------------------------
    axs = []
    plot_steps = [59, 47, 35, 23, 11, 0]
    for row in range(2):
        for col in range(1, 4):
            ax = fig.add_subplot(gs[row, col])
            t = plot_steps.pop()
            file = f"obs_imgs_DINO_1_6/frame_{t:03d}.png"
            ax.imshow(np.asarray(Image.open(file)))
            ax.set_title(f"step={t:d}")
            ax.axis("off")
            axs.append(ax)

    # Save the plot
    # plt.tight_layout()
    # file_path = "imgs/" + "turtlebot_rollout" + ".pdf"
    # plt.savefig(file_path, format="pdf", dpi=300, bbox_inches='tight')
    file_path = "imgs/" + "turtlebot_rollout" + ".png"
    plt.savefig(file_path, format="png", dpi=300, bbox_inches='tight')
    # plt.close()
    
