from dyn.quad_10d_of_perception import Quad10d_OF
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
    name = "Quad10d_OF_Perception"
    
    m = Quad10d_OF()

    # Problem setup
    Q = np.diag([1., 1., 1., 1., 1., 1., 1., 0., 0., 0.])
    R = np.diag([1., 1., 0.1])
    Qf = np.diag([50., 50., 50., 5., 5., 5., 5., 1., 1., 1.])

    N = 35 # horizon length

    m.dt = 0.15
    obstacles = [(np.array([1., 1.]), 0.2), (np.array([-1.5, 4.]), 0.2), (np.array([0.5, 3.]), 0.2),
                 (np.array([-0.5, 1.]), 0.2), (np.array([-1., 2.5]), 0.2)]

    # Setup solvers
    parallel = True
    verbose=False
    solver = SCP_OF_SLS(N, Q, R, m, Qf,
                        Q_reg=5e5 * np.eye(m.nx),
                        R_reg=5e5 * np.eye(m.nu),
                        Q_reg_f=5e5 * np.eye(m.nx),
                        H_reg_coef=1e-6,
                        parallel=parallel,
                        obstacles=obstacles,
                        certainty_equivalent=False,
                        non_robust=False,
                        verbose=verbose)

    # Disable verbose output for cleaner test
    solver.fast_SLS_solver.verbose = verbose
    solver.fast_SLS_solver.save_it_data = False
    
    x0_all = [
        np.array([0.9, 3.1, 1., 0.001, 0.001, 0.001, 0.001, -0.001, -0.001, -0.001]),
        np.array([-1.5, 1.5, 1., 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]),
        np.array([-1.5, 1., 1.4, 0.001, 0.001, 0.001, -0.001, -0.001, -0.001, -0.001]),
        np.array([-1.06593470, 1.5, 1.10456181, 2.67332447e-05, 3.83307696e-06, 
                7.08075505e-04, -7.57944103e-04, -6.61702048e-04, -3.65235395e-04, -6.55121415e-04]),
        np.array([1.10908462, 1.74590952, 1.23074318, 8.81476428e-04, -3.21073508e-04, 
                -2.24316359e-04, 3.01206361e-04, -9.99463201e-04, 3.67372965e-04, 8.55874892e-04]), 
        np.array([-1., 1.2, 1.43348860, -6.51474351e-04, -1.19435135e-05, 
                1.27502407e-04, 5.00949954e-04, -3.93577815e-04, -3.86666605e-04,  9.25646947e-04]),
        np.array([1.24921277, 3.30709532, 1.39855684, -3.54630626e-04, 9.60420333e-04, 
                -1.21189988e-04, 2.53206418e-04, 2.19917194e-04, -5.26161249e-06, -9.85192052e-04]),
        np.array([9.18497747e-01, 3.49854528, 1.43001580, -3.38818143e-04, 8.12694421e-04, 
                1.25370733e-04, 1.99184193e-04, 3.67443620e-05, -7.20104659e-04, 1.75409946e-04]),
        np.array([9.09025146e-01, 1.78647539, 1.01162208, 9.47650516e-04, -4.24438500e-04, 
                -3.27323338e-04, 1.52521899e-05, -9.33170630e-04, -8.26410964e-04, 1.50665656e-04]),
        np.array([1.38664083, 1.67795747, 1.33548519, 2.24796503e-04, 4.02040567e-04, 
                5.05786201e-04, -2.31956416e-05, 7.10199453e-04, -7.37747249e-04, -1.77197178e-04]),
        np.array([1.5, 1., 1.14684690, 8.68682345e-04, -1.83401133e-04, 
                -7.98845957e-04, -6.33092081e-04, 5.78230045e-05, -9.41358914e-07, 9.29638793e-04]),
        np.array([1.45267802, 1.19938593, 1.02210279, 9.47074739e-04, 7.81317063e-04, 
                7.47597111e-04, -5.36177642e-06, 2.49029350e-04, -2.07637014e-04, 7.84336561e-06]),
        np.array([1.43255604, 2.09522370, 1.17212090, 5.87481746e-05, 5.36093425e-04, 
                4.45792076e-04, -5.08720833e-05, 2.62424053e-04, -1.46258867e-04, 1.84814170e-04]),
        np.array([1.26462429, 2.70203864, 1.10651198, -3.24105474e-04, -9.03459809e-05, 
                2.85908747e-04, 1.31491574e-05, 5.16992626e-04, -3.39725758e-04, 7.16878499e-04]),
        np.array([1.01062306, 1.4, 1.41533893, -7.47929053e-04, 3.96140517e-04, 
                -3.42311859e-04, 8.37502304e-04, 5.13426549e-04, -4.52686696e-05, -2.36126558e-04]),
    ]
    xG_all = [
        np.array([-1.5, 2.5, 1.4, 0.001, 0.001, 0.001, -0.001, -0.001, -0.001, -0.001]),
        np.array([1., 3., 1.4, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]),
        np.array([1.5, 1., 1., 0.001, 0.001, 0.001, 0.001, -0.001, -0.001, -0.001]),
        np.array([1.61307398, 1.20354005, 1.24111589, 8.81952541e-04, -6.48578657e-04, 
                3.51944254e-04, 4.14436699e-05, 8.59153391e-04, -1.70811717e-04, -6.19871227e-04]),
        np.array([-1.34084882, 1.78725668, 1.20805872, 4.51291291e-04, 5.75905947e-04, 
                3.45811334e-04, 8.16050092e-04, 2.12146374e-04, -6.37836095e-04, -4.68739581e-04]),
        np.array([1.6, 1., 1.02158901, -2.85475923e-04, -2.36661149e-04, 
                9.50059027e-04, 9.52208643e-04, -8.45302816e-04, 1.31390656e-04, -6.67949121e-04]),
        np.array([-1.47097551, 1.34677568, 1.20403039, -7.30086075e-04, -8.70862525e-04, 
                7.07543827e-04, -1.94519710e-04, -1.80625561e-04, 5.53913815e-04, 9.10504329e-04]),
        np.array([-1.71537874, 2.44335244, 1.46160644, -4.84628964e-04, 7.14558262e-04, 
                9.38953421e-04, -2.79352447e-04, -6.87336201e-05, -5.84337307e-04, 9.77690832e-04]),
        np.array([-1.60851273, 2.16588722, 1.38097482, -4.07575821e-04, 4.00385190e-04, 
                3.07901248e-04, -8.74778461e-04, 2.45939679e-04, 8.91267063e-04, 8.47268430e-04]),
        np.array([-1.36737268, 2.88271015, 1.48649437, 2.02694831e-04, -6.57258878e-04, 
                -1.13491042e-04, 7.77860203e-04, 6.81344067e-04, 6.41901408e-04, 2.28578616e-04]),
        np.array([-1.29382094, 0.95, 1.10975137, -6.54304150e-04, 3.02708499e-04, 
                1.60718466e-04, -8.12299266e-04, -8.79364004e-04, -6.42443978e-04, 9.01851822e-04]),
        np.array([-1.39590546, 2.94564206, 1.49228515, -3.82756775e-04, 5.93724258e-04, 
                -7.61641806e-04, 6.09836073e-04, 5.10959898e-04, -4.97125913e-04, 2.34471366e-04]),
        np.array([-1.36468461, 2.55195862, 1.42169291, -1.27241989e-04, -2.78517746e-04, 
                5.04172499e-04, 7.38056544e-04, 7.46456515e-05, -8.69925008e-04, 9.81464019e-04]),
        np.array([-1.76436012, 2.87261109, 1.22441209, 1.02962855e-04, 1.23681998e-04, 
                -5.61430850e-04, 4.34758068e-04, -5.70585013e-04, -1.15270628e-04, -2.51720155e-05]),
        np.array([-1.50368540, 1., 1.12272043, -2.51225175e-05, 8.25710192e-06, 
                -8.76879956e-05, -8.43472772e-04, 1.55166766e-04, -9.48834834e-04, -2.95726656e-05]),
    ]
    idx = 0
    x0 = x0_all[idx]
    xG = xG_all[idx]
    
    start = time.perf_counter()
    solution = solver.solve(x0, xG, obstacles)
    end = time.perf_counter()
    print("total runtime: {:.3f}s".format(end-start))
    assert solution['success']
    # check_parameterization(solver, N, m.nx, m.nu, m.ny, solution, True)

    # Create a figure for plotting
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    Phi_xx_fast = solution['Phi_xx']
    Phi_xy_fast = solution['Phi_xy']
    X_fast = solution['primal_x']
    X_init = solution['initial_x']
    
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
    plt.tight_layout()
    os.makedirs("imgs", exist_ok=True)
    file_path = "imgs/" + name + ".png"
    plt.savefig(file_path, format="png", dpi=300, bbox_inches='tight')
    plt.close()

    os.makedirs("data", exist_ok=True)
    np.savez(f"data/quad10d_perception_{idx}.npz", 
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
