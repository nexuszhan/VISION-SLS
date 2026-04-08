import cvxpy as cp
import numpy as np
from scipy.linalg import block_diag

from util.SLS import SLS

def optimize_tube(m, N, A_list, B_list, C_list, E_list, F_list, 
                  Q_reg, R_reg, Q_reg_f, eta, eta_f):
    nx = m.nx
    nu = m.nu
    nw = m.nw
    ny = m.ny

    # System Response Matrix
    Phi_xx = [[cp.Variable((nx, nx), name='Phi_xx_{},{}'.format(i, j)) for j in range(N + 1)] for i in range(N + 1)]
    Phi_ux = [[cp.Variable((nu, nx), name='Phi_ux_{},{}'.format(i, j)) for j in range(N + 1)] for i in range(N + 1)]
    Phi_xy = [[cp.Variable((nx, ny), name='Phi_xy_{},{}'.format(i, j)) for j in range(N + 1)] for i in range(N + 1)]
    Phi_uy = [[cp.Variable((nu, ny), name='Phi_uy_{},{}'.format(i, j)) for j in range(N + 1)] for i in range(N + 1)]

    Phi_xx_mat = cp.bmat([[Phi_xx[i][j] for j in range(N + 1)] for i in range(N + 1)])
    Phi_ux_mat = cp.bmat([[Phi_ux[i][j] for j in range(N + 1)] for i in range(N + 1)])
    Phi_xy_mat = cp.bmat([[Phi_xy[i][j] for j in range(N + 1)] for i in range(N + 1)])
    Phi_uy_mat = cp.bmat([[Phi_uy[i][j] for j in range(N + 1)] for i in range(N + 1)])
    # last line of Phi_u_mat is not used ...

    # Define the cost function
    cost = 0

    # Add the system response matrix constraints in cost function
    # Q_reg = np.sqrt(Q_reg)
    # R_reg = np.sqrt(R_reg)
    # Q_reg_f = np.sqrt(Q_reg_f)

    # Q_reg_blk = block_diag(np.kron(np.eye(N), Q_reg), Q_reg_f)
    # R_reg_blk = np.kron(np.eye(N + 1), R_reg)
    # # concatenate the Phi matrices
    # Phi_mat = cp.hstack([cp.vstack([Phi_xx_mat, Phi_ux_mat]), cp.vstack([Phi_xy_mat, Phi_uy_mat])])

    # error_bounds = block_diag(*E_list)
    # error_bounds = block_diag(error_bounds, *F_list)
    
    # cost += cp.sum_squares(block_diag(Q_reg_blk, R_reg_blk) @ Phi_mat @ error_bounds)

    G = np.asarray(m.G)
    Gf = np.asarray(m.Gf)

    def quad_form_diag(G_mat, eta_vec):
        return G_mat.T @ (G_mat * eta_vec[:, None])
    
    # for jj in range(N):
    #     E_j = E_list[jj]
    #     F_j = F_list[jj]
    #     for kk in range(jj, N):
    #         Cost_k = quad_form_diag(G, eta[kk, jj])
    #         Q_reg_k = np.sqrt(Q_reg + Cost_k[:nx, :nx])
    #         R_reg_k = np.sqrt(R_reg + Cost_k[nx:, nx:])
    #         cost += cp.sum_squares(Q_reg_k @ Phi_xx[kk][jj] @ E_j)
    #         cost += cp.sum_squares(R_reg_k @ Phi_ux[kk][jj] @ E_j)
    #         cost += cp.sum_squares(Q_reg_k @ Phi_xy[kk][jj] @ F_j)
    #         cost += cp.sum_squares(R_reg_k @ Phi_uy[kk][jj] @ F_j)

    eta_arr = np.asarray(eta)

    def quad_form_diag_batch(G_mat, eta_mat):
        weighted = eta_mat[:, :, None] * G_mat[None, :, :]
        return np.einsum('pn,kpm->knm', G_mat, weighted, optimize=True)

    for jj in range(N):
        E_j = E_list[jj]
        F_j = F_list[jj]
        
        eta_kj = eta_arr[jj:N, jj]
        Cost_k = quad_form_diag_batch(G, eta_kj)
        Q_regs = np.sqrt(Q_reg[None, :, :] + Cost_k[:, :nx, :nx])
        R_regs = np.sqrt(R_reg[None, :, :] + Cost_k[:, nx:, nx:])
        Q_reg_blk = block_diag(*Q_regs)
        R_reg_blk = block_diag(*R_regs)
        Phi_xx_col = cp.vstack([Phi_xx[kk][jj] for kk in range(jj, N)])
        Phi_ux_col = cp.vstack([Phi_ux[kk][jj] for kk in range(jj, N)])
        Phi_xy_col = cp.vstack([Phi_xy[kk][jj] for kk in range(jj, N)])
        Phi_uy_col = cp.vstack([Phi_uy[kk][jj] for kk in range(jj, N)])
        cost += cp.sum_squares(Q_reg_blk @ Phi_xx_col @ E_j)
        cost += cp.sum_squares(R_reg_blk @ Phi_ux_col @ E_j)
        cost += cp.sum_squares(Q_reg_blk @ Phi_xy_col @ F_j)
        cost += cp.sum_squares(R_reg_blk @ Phi_uy_col @ F_j)

    for jj in range(N + 1):
        Cost_f = quad_form_diag(Gf, eta_f[jj])
        Q_reg_f_j = np.sqrt(Q_reg_f + Cost_f)
        cost += cp.sum_squares(Q_reg_f_j @ Phi_xx[N][jj] @ E_list[jj])
        cost += cp.sum_squares(Q_reg_f_j @ Phi_xy[N][jj] @ F_list[jj])

    constraints = []
    # Define the constraints
    A_blk = SLS.convert_list_to_blk_matrix(A_list)
    B_blk = SLS.convert_list_to_blk_matrix(B_list)
    C_blk = SLS.convert_list_to_blk_matrix(C_list)

    # add a block at the end of A_blk
    A_blk = block_diag(A_blk, np.zeros((nx, nx)))
    B_blk = block_diag(B_blk, np.zeros((nx, nu)))
    C_blk = block_diag(C_blk, np.zeros((ny, nx)))

    Z_downshift = SLS.get_block_downshift_matrix(N + 1, nx)
    Z_downshift_obs = SLS.get_block_downshift_matrix(N+1, ny)
    I_blk = np.eye(nx * (N + 1))
    Zeros_blk = np.zeros((nx * (N + 1), ny * (N + 1)))

    # system response matrix constraints
    constraints += [(I_blk - Z_downshift @ A_blk) @ Phi_xx_mat - Z_downshift @ B_blk @ Phi_ux_mat == I_blk]
    constraints += [(I_blk - Z_downshift @ A_blk) @ Phi_xy_mat - Z_downshift @ B_blk @ Phi_uy_mat == np.zeros((nx * (N + 1), ny * (N + 1)))]
    constraints += [Phi_xx_mat @ (
            I_blk - Z_downshift @ A_blk) - Phi_xy_mat @ Z_downshift_obs @ C_blk == I_blk] 
    constraints += [Phi_ux_mat @ (I_blk - Z_downshift @ A_blk) - Phi_uy_mat @ Z_downshift_obs @ C_blk == np.zeros((nu * (N + 1), nx * (N + 1)))]
    # last block of affine constraint are not used

    # add causal structure constraints
    for k in range(N):
        for j in range(N + 1):
            if j > k:
                constraints += [Phi_xx[k][j] == np.zeros((nx, nx))]
                constraints += [Phi_ux[k][j] == np.zeros((nu, nx))]
                constraints += [Phi_xy[k][j] == np.zeros((nx, ny))]
                constraints += [Phi_uy[k][j] == np.zeros((nu, ny))]

    # Define the problem
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.MOSEK, mosek_params={
        'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-5,
        'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-5,
        'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-5
        },
    verbose=False
    )

    Phi_xx_tensor= np.array([[
        Phi_xx[i][j].value for j in range(N + 1)] for i in range(N + 1)])
    Phi_ux_tensor = np.array([[
        Phi_ux[i][j].value for j in range(N + 1)] for i in range(N + 1)])
    Phi_xy_tensor = np.array([[
        Phi_xy[i][j].value for j in range(N + 1)] for i in range(N + 1)])
    Phi_uy_tensor = np.array([[
        Phi_uy[i][j].value for j in range(N + 1)] for i in range(N + 1)])

    return Phi_xx_tensor, Phi_ux_tensor, Phi_xy_tensor, Phi_uy_tensor, cost.value
