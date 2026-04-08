from dyn.G1.G1_of import G1OF
import numpy as np
from solver.SCP_OF_SLS import SCP_OF_SLS
from datetime import datetime
import os
import time
from matplotlib import pyplot as plt
from util.footnote import add_footnote_time

from util.plot import *
from util.sanity_checks import *

if __name__ == '__main__':
    name = "G1_OF"

    m = G1OF()

    Q = np.diag(
            [0.0, 100.0, 0.0]  # x, y, z
            + [10.0, 100.0, 10.0, 100.0]  # axis-angle residual rotation s.t. R = R_ref * r
            + [75.0] * (m.nq - 7)  # joint positions
            + [0.0, 100.0, 0.0]  # base linear velocities
            + [100.0, 100.0, 100.0]  # base angular velocities
            + [0.5] * (m.ndq - 6)  # joint velocities
        )
    R = np.eye(m.nu) * 1e-5
    Qf = np.diag(
                [0.0, 100.0, 50.0]  # x, y, z
                + [100.0] * 4  # axis-angle residual rotation s.t. R = R_ref * r
                + [100.0] * (m.nq - 7)  # joint positions
                + [0.0, 100.0, 50.0]  # base linear velocities
                + [100.0, 100.0, 100.0]  # base angular velocities
                + [0.0] * (m.ndq - 6)  # joint velocities
                )

    data1 = np.load("data/G1_4.npz") 
    primal_x = data1["initial_traj"]
    primal_u = data1["initial_input"]
    x0 = primal_x[:,0]

    data2 = np.load("data/reference_traj_G1_extracted.npz")
    xG = data2["traj"]

    N = primal_x.shape[1] - 1

    # Setup solvers
    parallel = True
    verbose = False
    solver = SCP_OF_SLS(N, Q, R, m, Qf,
                        Q_reg=1e5 * np.eye(m.nx),
                        R_reg=1e5 * np.eye(m.nu),
                        Q_reg_f=1e5 * np.eye(m.nx),
                        certainty_equivalent=True,
                        parallel=parallel,
                        non_robust=False,
                        verbose=verbose
                        initialization_file="data/G1_4.npz")

    # Disable verbose output for cleaner test
    solver.fast_SLS_solver.verbose = verbose
    solver.fast_SLS_solver.save_it_data = False
    
    start = time.perf_counter()
    solution = solver.solve(x0, xG)
    end = time.perf_counter()
    # assert scp_solution['success']

    np.savez("data/G1_robust.npz", 
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
    