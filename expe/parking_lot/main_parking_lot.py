from dyn.unicycle_of_perception import Unicycle_OF
import numpy as np
from solver.SCP_OF_SLS import SCP_OF_SLS
from datetime import datetime
import os
import time
from matplotlib import pyplot as plt
from util.footnote import add_footnote_time

from util.plot import *
from util.sanity_checks import *
import time

import cProfile
from util.profile import show_func_runtimes

def main():
    name = "Unicycle_OF_Perception"
    
    m = Unicycle_OF()

    # Problem setup
    Q = np.eye(m.nx)
    R = np.eye(m.nu) * 0.5
    Qf = 100 * np.eye(m.nx) 
    Q[2,2] = 0
    Q[3,3] = 0
    Qf[2,2] = 0
    Qf[3,3] = 10.
    N = 30 # horizon length

    m.dt = 0.15
    x_max = np.array([0.5, 1.9, 2*np.pi, 2.]) # x, y, theta, v
    x_min = np.array([-3.5, -2., -2*np.pi, -1.])
    u_max = np.array([np.pi, 4.]) # angular_vel, linear acceleration
    u_min = np.array([-np.pi, -4.])
    m.g = np.concatenate((x_max, u_max, -x_min, -u_min))

    x_max_f0 = np.array([0.45, 0.65, 2*np.pi, 1.]) # x, y, theta, v
    x_min_f0 = np.array([0., 0.35, -2*np.pi, -1.])
    x_max_f1 = np.array([0.2, 0.2, 2*np.pi, 1.]) # 1-6, 14-
    x_min_f1 = np.array([-0.2, -0.2, -2*np.pi, -1.])
    x_max_f2 = np.array([0.2, 1.7, 2*np.pi, 1.]) # 7-10
    x_min_f2 = np.array([-0.2, 1.3, -2*np.pi, -1.])
    x_max_f3 = np.array([0.2, -1.05, 2*np.pi, 1.]) # 11-13
    x_min_f3 = np.array([-0.2, -1.45, -2*np.pi, -1.])

    x_min_f_all = [x_min_f0] + [x_min_f1]*6 + [x_min_f2]*4 + [x_min_f3]*3 + [x_min_f1]
    x_max_f_all = [x_max_f0] + [x_max_f1]*6 + [x_max_f2]*4 + [x_max_f3]*3 + [x_max_f1]

    # x_max_f = np.array([0.45, 0.65, 2*np.pi, 1.]) # x, y, theta, v
    # x_min_f = np.array([0., 0.35, -2*np.pi, -1.])
    # x_max_f = np.array([0.2, 0.2, 2*np.pi, 1.]) # 1-6, 14-
    # x_min_f = np.array([-0.2, -0.2, -2*np.pi, -1.])
    # x_max_f = np.array([0.2, 1.7, 2*np.pi, 1.]) # 7-10
    # x_min_f = np.array([-0.2, 1.3, -2*np.pi, -1.])
    # x_max_f = np.array([0.2, -1.05, 2*np.pi, 1.]) # 11-13
    # x_min_f = np.array([-0.2, -1.45, -2*np.pi, -1.])
    # m.gf = np.concatenate((x_max_f, -x_min_f))

    x0_all = [np.array([-2.1, -1.75, np.pi/2, 0.]), np.array([-2.5, -1.5, 0., 0.]), np.array([-2.5, -1., 0., 0.]),
            np.array([-2., 1.5, 0., 0.]), np.array([-1.25, 1.2, 0., 0.]), np.array([-0.75, 1.75, -np.pi/2, 0.]),
            np.array([-1.5, -1.75, np.pi/2, 0.]), np.array([-2.5, 0., np.pi*0.35, 0.]), np.array([-2.5, 0.5, np.pi/4, 0.]),
            np.array([-2.5, 1., 0., 0.]), np.array([-2.75, 1.25, np.pi/4, 0.]), np.array([-2., -1.5, 0., 0.]),
            np.array([-2.5, -1.5, 0., 0.]), np.array([-2.25, -0.75, 0., 0.]), np.array([-2.2, -0.6, 0.6, 0.])]
    xG_0 = np.array([0.25, 0.5, 0., 0.])
    xG_1 = np.array([0., 0., 0., 0.])
    xG_2 = np.array([0., 1.5, 0., 0.])
    xG_3 = np.array([0., -1.25, 0., 0.])
    xG_all = [xG_0] + [xG_1]*6 + [xG_2]*4 + [xG_3]*3 + [xG_1]

    eval_idx = 0
    x0 = x0_all[eval_idx]
    xG = xG_all[eval_idx]
    x_min_f = x_min_f_all[eval_idx]
    x_max_f = x_max_f_all[eval_idx]
    m.gf = np.concatenate((x_max_f, -x_min_f))

    obstacles = [(np.array([-3.55, 0.7]), 0.4), (np.array([-0.6, 1.15]), 0.35), (np.array([-1.2, -1.2]), 0.4)]
    # Setup solvers
    parallel = True
    verbose = False
    solver = SCP_OF_SLS(N, Q, R, m, Qf,
                             Q_reg=1e5 * np.eye(m.nx),
                             R_reg=1e5 * np.eye(m.nu),
                             Q_reg_f=3e6 * np.eye(m.nx),
                             H_reg_coef=4.,
                             parallel=parallel,
                             obstacles=obstacles,
                             certainty_equivalent=False,
                             non_robust=False,
                             verbose=verbose)

    # Disable verbose output for cleaner test
    solver.fast_SLS_solver.verbose = verbose
    solver.fast_SLS_solver.save_it_data = False
    
    start = time.perf_counter()
    solution = solver.solve(x0, xG, obstacles)
    end = time.perf_counter()
    print("total runtime: {:.3f}s".format(end-start))
    assert solution['success']

    # Create a figure for plotting
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    Phi_xx_fast = solution['Phi_xx']
    Phi_xy_fast = solution['Phi_xy']
    X_fast = solution['primal_x']
    X_init = solution['initial_x']
    # check_parameterization(scp_sls_solver, N, m.nx, m.nu, m.ny, solution_fast_sls, delay)
    
    cf = m.plot_disturbance_map(ax)
    cbar = fig.colorbar(cf, ax=ax)

    import matplotlib.patches as patch
    for obstacle in obstacles:
        p = patch.Circle(obstacle[0], obstacle[1])
        ax.add_patch(p)

    for i in range(solution['primal_x'].shape[1]):
        # plot rectangle based on backoff
        backoff_fast_sls = solution['backoff_x'][i, :]
        m.plot_tube_as_rectangle(backoff_fast_sls, X_fast[:, i], ax=ax)

        backoff_initial = solution["initial_backoff_x"][i,:]
        m.plot_tube_as_rectangle(backoff_initial, X_init[:,i], ax=ax)

    # Plot nominal trajectory
    m.plot_nominal_trajectory(X_fast, ax=ax)
    ax.plot(X_init[0, :], X_init[1, :], 'b-', linewidth=1, label='Initial trajectory')
    ax.legend()
    
    ax.set_title("SCP-OF-SLS")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.set_aspect('equal', adjustable='datalim')
    # Add overall title
    fig.suptitle(name, fontsize=14)
    # Add footnote with timestamp
    add_footnote_time(plt)

    # Save the plot
    os.makedirs("imgs", exist_ok=True)
    plt.tight_layout()
    file_path = "imgs/" + name + ".png" 
    plt.savefig(file_path, format="png", dpi=300, bbox_inches='tight') 
    plt.close()

    os.makedirs("data", exist_ok=True)
    np.savez(f"data/unicycle_of_perception_{eval_idx}.npz", 
         nominal_traj=np.array(solution["primal_x"]),
         nominal_input=np.array(solution["primal_u"]),
         initial_traj=np.array(solution["initial_x"]),
         initial_backoff_x=np.array(solution["initial_backoff_x"]),
         Phi_xx=np.array(solution["Phi_xx_mat"]),
         Phi_ux=np.array(solution["Phi_ux_mat"]),
         Phi_xy=np.array(solution["Phi_xy_mat"]),
         Phi_uy=np.array(solution["Phi_uy_mat"]),
         backoff=np.array(solution["backoff"]),
         backoff_x=np.array(solution["backoff_x"]),
         backoff_f=np.array(solution["backoff_f"])
    )

if __name__ == "__main__":
    show_profile = False
    profile_file_name = "prof.out"

    if show_profile:
        cProfile.run("main()", profile_file_name)
        show_func_runtimes(profile_file_name)
    else:
        main()
