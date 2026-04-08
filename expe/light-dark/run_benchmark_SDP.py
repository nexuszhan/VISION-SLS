import numpy as np
import casadi as ca
from matplotlib import pyplot as plt
import time, os

from solver.SCP_OF_SLS_SDP import SCP_OF_SLS
from dyn.light_dark_of import LightDark_OF

import cProfile
from util.profile import show_func_runtimes_SDP

def main():
     # Setup white-dark
    m = LightDark_OF()

    # Problem setup
    Q = np.eye(m.nx)
    R = np.eye(m.nu)
    Qf = 50*np.eye(m.nx) 
    N = 20 # horizon length

    # Setup solvers
    solver = SCP_OF_SLS(N, Q, R, m, Qf,
                        Q_reg=1e4 * np.eye(m.nx),
                        R_reg=1e4 * np.eye(m.nu),
                        Q_reg_f=1e4 * np.eye(m.nx),
                        H_reg_coef=1e-6)

    # Disable verbose output for cleaner test
    solver.fast_SLS_solver.verbose = False
    solver.fast_SLS_solver.save_it_data = False
    
    start = time.perf_counter()
    x0 = np.array([0, 2])  # Initial condition
    solution = solver.solve(x0)
    end = time.perf_counter()
    print("total runtime: {:.3f}".format(end-start))
    print("tube cost: {:.3f}".format(solution["cost_tube"]))
    # assert scp_solution['success']

    # Create a figure for plotting
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    Phi_xx_fast = solution['Phi_xx']
    Phi_xy_fast = solution['Phi_xy']
    X_fast = solution['primal_x']
    X_init = solution['initial_x']

    # visualize disturbance 
    cf = m.plot_disturbance_map(ax)
    cbar = fig.colorbar(cf, ax=ax)

    for i in range(solution['primal_x'].shape[1]):
        # plot rectangle based on backoff
        backoff_fast_sls = solution['backoff_x'][i, :]
        m.plot_tube_as_rectangle(backoff_fast_sls, X_fast[:, i], ax=ax)

        backoff_initial = solution["initial_backoff_x"][i,:]
        m.plot_tube_as_rectangle(backoff_initial, X_init[:,i], ax=ax)

    # Plot nominal trajectory
    m.plot_nominal_trajectory(X_fast, ax=ax)
    ax.plot(X_init[0, :], X_init[1, :], 'b-', linewidth=1, label='Initial trajectory')
    ax.plot(X_init[0, -1], X_init[1, -1], 'bo', markersize=4, label='Initial Final state')
    ax.legend()

    ax.set_title("SCP-OF-SLS")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.set_aspect('equal', adjustable='datalim')

    # Save the plot
    plt.tight_layout()

    os.makedirs("imgs", exist_ok=True)
    file_path = "imgs/light-dark_SDP.png" # 660, 832
    plt.savefig(file_path, format="png", dpi=300, bbox_inches='tight')
    plt.close()

    os.makedirs("data", exist_ok=True)
    np.savez("data/light-dark_SDP.npz", 
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

    delay = True
    primal_x = solution["primal_x"]
    primal_u = solution["primal_u"]
    Phi_xx = solution["Phi_xx_mat"]
    Phi_ux = solution["Phi_ux_mat"]
    Phi_xy = solution["Phi_xy_mat"]
    Phi_uy = solution["Phi_uy_mat"]
    tube = solution["backoff"]
    tube_f = solution["backoff_f"]
    K_mat = Phi_uy - Phi_ux @ np.linalg.inv(Phi_xx) @ Phi_xy

    ROLLOUT_NUM = 10
    rollout_states = np.zeros((ROLLOUT_NUM, m.nx, N+1))
    rollout_states_open = np.zeros((ROLLOUT_NUM, m.nx, N+1))
    feedback_error = 0.
    feedback_error_terminal = 0.
    openloop_error = 0.
    openloop_error_terminal = 0.
    np.random.seed(42)

    for rollout in range(ROLLOUT_NUM):
        print("rollout"+str(rollout))
        states = np.zeros((m.nx, N+1))
        states[:,0] = x0
        observations = np.zeros((m.ny, N+1))

        disturbance_state = np.random.random((m.nx, N)) - 0.5
        disturbance_state = disturbance_state / np.linalg.norm(disturbance_state, axis=0, keepdims=True)
        disturbance_obs = np.random.random((m.ny, N)) - 0.5
        disturbance_obs = disturbance_obs / np.linalg.norm(disturbance_obs, axis=0, keepdims=True)
        
        for t in range(1, N+1):
            delta_u = np.zeros((m.nu))
            if delay:
                # TODO: add a x(-1) to compute y(0) in this case
                if t == 1:
                    observations[:,t-1] = np.squeeze(m.measurement(states[:,t-1], disturbance_obs[:,t-1]))
                else:
                    observations[:,t-1] = np.squeeze(m.measurement(states[:,t-2], disturbance_obs[:,t-1]))
                
                delta_u += K_mat[(t-1)*m.nu:t*m.nu, 0:m.ny] @ (observations[:,0] - m.C @ (primal_x[:,0]))
                for j in range(1, t):
                    delta_u += K_mat[(t-1)*m.nu:t*m.nu, j*m.ny:(j+1)*m.ny] @ (observations[:,j] - m.C @ (primal_x[:,j-1]))
            else: # unused
                observations[:,t-1] = m.measurement(states[:,t-1], disturbance_obs[:,t-1])
                for j in range(t):
                    delta_u += K_mat[(t-1)*m.nu:t*m.nu, j*m.ny:(j+1)*m.ny] @ (observations[:,j] - m.C @ primal_x[:,j])
            
            u = primal_u[:,t-1] + delta_u

            states[:,t] = np.squeeze(m.ddyn(states[:,t-1], u, disturbance_state[:,t-1], m.dt))

            feedback_error += np.linalg.norm(states[:,t] - primal_x[:,t])
            # plt.imsave(f"video_frames/frame_{t-1:03d}.png", rgb_im)
        rollout_states[rollout, :, :] = states.copy()
        feedback_error_terminal += np.linalg.norm(states[:,-1] - primal_x[:,-1])
        
        disturbance_state_open = disturbance_state.copy()  # NEW

        states_open = np.zeros((m.nx, N + 1))
        states_open[:, 0] = x0
        for t in range(1, N + 1):
            u_ol = primal_u[:, t - 1]  # scheduled input only
            states_open[:, t] = np.squeeze(
                m.ddyn(states_open[:, t - 1], u_ol, disturbance_state_open[:, t - 1], m.dt)
            )

            openloop_error += np.linalg.norm(states_open[:,t] - primal_x[:,t])
        openloop_error_terminal += np.linalg.norm(states_open[:,-1] - primal_x[:,-1])
        rollout_states_open[rollout, ...] = states_open.copy()

    np.savez("data/rollout_SDP.npz", 
            rollout_states=rollout_states,
            rollout_states_open=rollout_states_open,
            feedback_error=feedback_error,
            feedback_error_terminal=feedback_error_terminal,
            openloop_error=openloop_error,
            openloop_error_terminal=openloop_error_terminal
            )

if __name__ == "__main__":
    show_profile = False
    profile_file_name = "prof_SDP.out"
    
    if show_profile:
        cProfile.run("main()", profile_file_name)
        show_func_runtimes_SDP(profile_file_name)
    else:
        main()