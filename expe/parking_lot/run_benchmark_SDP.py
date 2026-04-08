import numpy as np
import casadi as ca
from matplotlib import pyplot as plt
import time, os

from solver.SCP_OF_SLS_SDP import SCP_OF_SLS
from dyn.unicycle_of_perception import Unicycle_OF
from util.footnote import add_footnote_time

import cProfile, pstats
from util.profile import show_func_runtimes_benchmark

def main():
    name = "Unicycle_OF_Perception"
    
    m = Unicycle_OF()

    # Problem setup
    Q = np.eye(m.nx)
    R = np.eye(m.nu) * 0.5
    Qf = 10 * np.eye(m.nx) 
    Q[2,2] = 0
    Q[3,3] = 0
    Qf[2,2] = 0
    N = 30 # horizon length

    m.dt = 0.15
    # x_max = np.array([10, 10, 10, 2.]) # x, y, theta, v
    # x_min = np.array([-10, -10, -10, -1.])
    x_max = np.array([0.5, 2., 2*np.pi, 2.]) # x, y, theta, v
    x_min = np.array([-3.5, -2., -2*np.pi, -1.])
    u_max = np.array([np.pi, 4.]) # angular_vel, linear acceleration
    u_min = np.array([-np.pi, -4.])
    m.g = np.concatenate((x_max, u_max, -x_min, -u_min))
    # x_max_f = np.array([0.75, 0.75, 2*np.pi, 1.]) # x, y, theta, v
    # x_min_f = np.array([-0.75, -0.75, -2*np.pi, -1.])
    x_max_f = np.array([0.25, 0.25, 2*np.pi, 1.]) # x, y, theta, v
    x_min_f = np.array([-0.25, -0.25, -2*np.pi, -1.])
    m.gf = np.concatenate((x_max_f, -x_min_f))
    
    # obs_centers = np.array([[-0.55, 0.7], [2.45, 1.25], [1.8, -1.2]]).T#[]
    # obs_radii = np.array([0.4, 0.35, 0.4])#[]
    obstacles = [(np.array([-3.55, 0.7]), 0.4), (np.array([-0.55, 1.25]), 0.35), (np.array([-1.2, -1.2]), 0.4)]

    # Setup solvers
    delay = True
    scp_sls_solver = SCP_OF_SLS(N, Q, R, m, Qf,
                             Q_reg=1e5 * np.eye(m.nx),
                             R_reg=1e5 * np.eye(m.nu),
                             Q_reg_f=1e5 * np.eye(m.nx),
                             Q_reg_obs=4e5 * np.eye(m.nx),
                             R_reg_obs=4e5 * np.eye(m.nu),
                             Q_reg_f_obs=4e5 * np.eye(m.nx),
                             delay=delay,
                             obstacles=obstacles,
                             state_dependent=True)

    # Disable verbose output for cleaner test
    scp_sls_solver.fast_SLS_solver.verbose = False
    scp_sls_solver.fast_SLS_solver.save_it_data = False

    # x0 = np.array([-2.5, 0.6, 0., 0.])  # Initial condition
    x0 = np.array([-1.7, 0.6, 0., 0.])  # Initial condition
    x0 = np.array([-2.1, -1.75, 0., 0.])  # Initial condition
    # x0 = np.array([-2., -1.6, 0., 0.])
    # xG = np.array([0., 0., 0., 0.])
    xG = np.array([0.5, 0.5, 0., 0.])
    scp_solution = scp_sls_solver.solve(x0, xG, obstacles)
    # assert scp_solution['success']

    # Create a figure for plotting
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    Phi_xx_fast = scp_solution['Phi_xx']
    Phi_xy_fast = scp_solution['Phi_xy']
    X_fast = scp_solution['primal_x']
    X_init = scp_solution['initial_x']
    # check_parameterization(scp_sls_solver, N, m.nx, m.nu, m.ny, solution_fast_sls, delay)
    
    cf = m.plot_disturbance_map(ax)
    cbar = fig.colorbar(cf, ax=ax)

    import matplotlib.patches as patch
    for obstacle in obstacles:
        p = patch.Circle(obstacle[0], obstacle[1])
        ax.add_patch(p)

    for i in range(scp_solution['primal_x'].shape[1]):
        # plot rectangle based on backoff
        backoff_fast_sls = scp_solution['backoff_x'][i, :]
        m.plot_tube_as_rectangle(backoff_fast_sls, X_fast[:, i], ax=ax)

        backoff_initial = scp_solution["initial_backoff_x"][i,:]
        m.plot_tube_as_rectangle(backoff_initial, X_init[:,i], ax=ax)

    # Plot nominal trajectory
    m.plot_nominal_trajectory(X_fast, ax=ax)
    ax.plot(X_init[0, :], X_init[1, :], 'b-', linewidth=1, label='Initial trajectory')
    # ax.plot(X_init[0, -1], X_init[1, -1], 'bo', markersize=4, label='Initial Final state')
    ax.legend()
    
    # print(scp_solution["cost_tube_K0"])

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

    np.savez("data/unicycle_of_perception_SDP.npz", 
         nominal_traj=np.array(scp_solution["primal_x"]),
         nominal_input=np.array(scp_solution["primal_u"]),
         initial_traj=np.array(scp_solution["initial_x"]),
         initial_backoff_x=np.array(scp_solution["initial_backoff_x"]),
         Phi_xx=np.array(scp_solution["Phi_xx_mat"]),
         Phi_ux=np.array(scp_solution["Phi_ux_mat"]),
         Phi_xy=np.array(scp_solution["Phi_xy_mat"]),
         Phi_uy=np.array(scp_solution["Phi_uy_mat"]),
         backoff=np.array(scp_solution["backoff"]),
         backoff_x=np.array(scp_solution["backoff_x"]),
         backoff_f=np.array(scp_solution["backoff_f"])
    )

if __name__ == "__main__":
    show_profile = False
    profile_file_name = "prof_benchmark.out"
    
    if show_profile:
        cProfile.run("main()", profile_file_name)
        show_func_runtimes_benchmark(profile_file_name)
    else:
        main()