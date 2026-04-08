from dyn.turtlebot_of import Turtlebot_OF
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

if __name__ == '__main__':
    name = "Turtlebot_OF_Perception"
    
    m = Turtlebot_OF()

    # Problem setup
    Q = np.diag([1., 1., 0., 0.1, 0.1])
    R = np.eye(m.nu) * 0.2
    Qf = np.diag([10., 10., 1., 1., 1.])
    N = 60 # horizon length

    m.dt = 0.2

    robot_radii = 0.2 
    obstacles = [(np.array([0.745, 0.656]), 0.432+robot_radii), 
                 (np.array([2.66, 0.72]), 0.254+robot_radii)]

    # Setup solvers
    parallel = True
    solver = SCP_OF_SLS(N, Q, R, m, Qf,
                        Q_reg=5e6 * np.eye(m.nx),
                        R_reg=5e6 * np.eye(m.nu),
                        Q_reg_f=5e6 * np.eye(m.nx),
                    #  H_reg_coef=1e-6, # didn't use this in harware experiment
                        parallel=parallel,
                        obstacles=obstacles,
                        init_guess=None,
                        certainty_equivalent=True,
                        non_robust=False) 

    # Disable verbose output for cleaner test
    solver.fast_SLS_solver.verbose = False
    solver.fast_SLS_solver.save_it_data = False

    x0 = np.array([0., 0., 0., 0., 0.]) 
    xG = np.array([3., 1.5, 0., 0., 0.]) 

    start = time.perf_counter()
    solution = solver.solve(x0, xG, obstacles)
    end = time.perf_counter()
    print("total runtime: {:.3f}".format(end-start))

    # Create a figure for plotting
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    Phi_xx_fast = solution['Phi_xx']
    Phi_xy_fast = solution['Phi_xy']
    X_fast = solution['primal_x']
    X_init = solution['initial_x']
    # check_parameterization(scp_sls_solver, N, m.nx, m.nu, m.ny, solution_fast_sls, delay)

    import matplotlib.patches as patch
    for obstacle in obstacles:
        p = patch.Circle(obstacle[0], obstacle[1], alpha=0.8)
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
    plt.tight_layout()
    file_path = "imgs/" + name + ".png"
    plt.savefig(file_path, format="png", dpi=300, bbox_inches='tight')
    plt.close()
    
    np.savez("Turtlebot_OF_DINO.npz", 
         nominal_traj=np.array(solution["primal_x"]),
         nominal_input=np.array(solution["primal_u"]),
         initial_traj=np.array(solution["initial_x"]),
         initial_input=np.array(solution["initial_u"]),
         initial_backoff_x=np.array(solution["initial_backoff_x"]),
         Phi_xx=np.array(solution["Phi_xx_mat"]),
         Phi_ux=np.array(solution["Phi_ux_mat"]),
         Phi_xy=np.array(solution["Phi_xy_mat"]),
         Phi_uy=np.array(solution["Phi_uy_mat"]),
         backoff=np.array(solution["backoff"]),
         backoff_x=np.array(solution["backoff_x"]),
         backoff_f=np.array(solution["backoff_f"])
    )