from numba import njit, prange
import numpy as np
import casadi as ca
import time
import os

fast_math = True

def np64_contig(x):
    # Scalars
    if np.isscalar(x):
        return np.float64(x)

    # CasADi matrices
    if isinstance(x, (ca.DM, ca.SX, ca.MX)):
        return np.ascontiguousarray(x.full(), dtype=np.float64)

    # NumPy arrays
    if isinstance(x, np.ndarray):
        if x.dtype == np.float64 and x.flags.c_contiguous:
            return x
        return np.ascontiguousarray(x, dtype=np.float64)

    # Lists / tuples (but prefer pre-stacking outside)
    if isinstance(x, (list, tuple)):
        return np.ascontiguousarray(
            np.array([np64_contig(xx) for xx in x], dtype=np.float64)
        )

    raise TypeError(f"Unsupported type for numba boundary: {type(x)}")

@njit(cache=True, fastmath=fast_math)
def riccati_step(A, B, Cx, Cu, Sk):
    x = B.T @ Sk
    y = A.T @ Sk
    K = -np.linalg.solve(Cu + x @ B, x @ A)
    S = Cx + y @ A + y @ B @ K
    return K, S

@njit(cache=True, fastmath=fast_math)
def quad_form_diag(G, eta_vec):
    weighted = G * eta_vec[:, None]
    return G.T @ weighted

@njit(cache=True, parallel=True, fastmath=fast_math)
def get_state_feedback(K_feedback, S_feedback, N, A, B, eta, eta_f, G, G_f, nx, Q_reg, R_reg, Q_reg_f):
    for jj in prange(N + 1):
        Cost_f = quad_form_diag(G_f, eta_f[jj])
        S_feedback[N, jj, :, :] = Q_reg_f + Cost_f

    for jj in prange(N):                 
        for kk in range(N-1, jj-1, -1):   
            Cost_k = quad_form_diag(G, eta[kk, jj])
            Cost_k_xx = Cost_k[:nx, :nx] + Q_reg
            Cost_k_uu = Cost_k[nx:, nx:] + R_reg
            K_feedback[kk, jj, :, :], S_feedback[kk, jj, :, :] = \
                riccati_step(A[kk], B[kk], Cost_k_xx, Cost_k_uu, S_feedback[kk+1, jj])

    return K_feedback, S_feedback

@njit(cache=True, parallel=True, fastmath=fast_math)
def state_forward_propagate(Phi_x_f, Phi_u_f, K_feedback, A, B, N, I):
    for jj in prange(N + 1):
        Phi_x_f[jj, jj, :, :] = I
        for kk in range(jj, N):
            Phi_u_f[kk, jj] = K_feedback[kk, jj] @ Phi_x_f[kk, jj]

            # close loop dynamics
            A_cl = A[kk] + B[kk] @ K_feedback[kk, jj]
            Phi_x_f[kk + 1, jj] = A_cl @ Phi_x_f[kk, jj]

    return Phi_x_f, Phi_u_f

def generate_state_Phi(N, nx, nu, A, B, Q_reg, R_reg, Q_reg_f, eta_x, eta_x_f, G, Gf):
    S_feedback_par = np.ascontiguousarray(np.zeros((N + 1, N + 1, nx, nx)))
    K_feedback_par = np.ascontiguousarray(np.zeros((N, N + 1, nu, nx)))
    
    eta_np = np64_contig(eta_x)     
    eta_f_np = np64_contig(eta_x_f)

    Phi_x_f_par = np.ascontiguousarray(np.zeros((N + 1, N + 1, nx, nx)))
    Phi_u_f_par = np.ascontiguousarray(np.zeros((N + 1, N + 1, nu, nx)))
    I = np.ascontiguousarray(np.eye(nx))

    K_feedback_par, S_feedback_par = get_state_feedback(K_feedback_par, S_feedback_par, N, A, B, eta_np, eta_f_np,
                                                    G, Gf, nx, Q_reg, R_reg, Q_reg_f)

    Phi_x_f_par, Phi_u_f_par = state_forward_propagate(Phi_x_f_par, Phi_u_f_par, K_feedback_par, 
                                                       A, B, N, I)

    return Phi_x_f_par, Phi_u_f_par, K_feedback_par, S_feedback_par

@njit(cache=True, parallel=True, fastmath=fast_math)
def generate_observer_feedback(L_obs, S_obs, N, A, C, E, F):
    E0Et = E[0] @ E[0].T
    for jj in prange(N + 1):
        S_obs[jj, 0] = E0Et 

    for jj in prange(1, N+1):
        for kk in range(1, jj+1):
            L_obs[jj, kk, :, :], S_obs[jj, kk, :, :] \
                = riccati_step(A[kk-1].T, C[kk-1].T, E[kk]@E[kk].T, F[kk]@F[kk].T, S_obs[jj, kk-1])
            
    return L_obs, S_obs

@njit(cache=True, parallel=True, fastmath=fast_math)
def observer_forward_propagate(Phi_x_o, Phi_y_o, L_obs, A, C, N, I):
    for jj in prange(N + 1):
        Phi_x_o[jj, jj] = I
        for kk in range(jj, 0, -1):
            A_cl = A[kk - 1] + L_obs[jj, kk].T @ C[kk - 1]
            Phi_x_o[jj, kk - 1] = Phi_x_o[jj, kk] @ A_cl
            Phi_y_o[jj, kk] = Phi_x_o[jj, kk] @ L_obs[jj, kk].T

    return Phi_x_o, Phi_y_o

def generate_observer_Phi(N, nx, ny, A, C, E, F,):
    Phi_y_o_par = np.ascontiguousarray(np.zeros((N + 1, N + 1, nx, ny)))
    Phi_x_o_par = np.ascontiguousarray(np.zeros((N + 1, N + 1, nx, nx)))
    S_obs_par = np.ascontiguousarray(np.zeros((N + 1, N + 1, nx, nx)))
    L_obs_par = np.ascontiguousarray(np.zeros((N + 1, N + 1, ny, nx)))
    I = np.ascontiguousarray(np.eye(nx))
    
    L_obs_par, S_obs_par = generate_observer_feedback(L_obs_par, S_obs_par, N, A, C, E, F)
    Phi_x_o_par, Phi_y_o_par = observer_forward_propagate(Phi_x_o_par, Phi_y_o_par, L_obs_par, A, C, N, I)

    return Phi_x_o_par, Phi_y_o_par, L_obs_par, S_obs_par

def comp_seq_par(N, seq1, par1, seq2, par2):
    for jj in range(N):
        for kk in range(jj+1):
            assert np.allclose(seq1[jj,kk], par1[jj,kk], atol=1e-5)
            assert np.allclose(seq2[jj,kk], par2[jj,kk], atol=1e-5)

@njit(cache=True, parallel=True, fastmath=fast_math)
def scan_tightening(N, ni, ni_f, Phi_xx, Phi_ux, Phi_xy, Phi_uy, E, F, G, Gf):
    # initialize the backoff
    beta = np.zeros((N, N, ni))  # backoff
    beta_f = np.zeros((N + 1, ni_f))

    sqrt_beta = np.zeros((N, N, ni))
    sqrt_beta_f = np.zeros((N + 1, ni_f))

    sqrt_beta_x = np.zeros((N, N, ni))
    sqrt_beta_x_f = np.zeros((N + 1, ni_f))
    sqrt_beta_y = np.zeros((N, N, ni))
    sqrt_beta_y_f = np.zeros((N + 1, ni_f))

    nx = Phi_xx.shape[2]
    Gx = np.ascontiguousarray(G[:, :nx])
    Gu = np.ascontiguousarray(G[:, nx:])

    # calculate the backoff
    # column-wise
    for jj in prange(N + 1):
        Ejj = E[jj]
        Fjj = F[jj]
        # row-wise
        for kk in range(jj, N):
            # evaluate the 2-norm of each component
            mat_1 = (Gx @ Phi_xx[kk, jj] + Gu @ Phi_ux[kk, jj]) @ Ejj
            mat_2 = (Gx @ Phi_xy[kk, jj] + Gu @ Phi_uy[kk, jj]) @ Fjj

            sqrt_beta_x[kk, jj, :] = np.sqrt(np.sum(mat_1**2, axis=1))
            sqrt_beta_y[kk, jj, :] = np.sqrt(np.sum(mat_2**2, axis=1))
            beta[kk, jj, :] = sqrt_beta_x[kk, jj, :] ** 2 + sqrt_beta_y[kk, jj, :] ** 2
            sqrt_beta[kk, jj, :] = sqrt_beta_x[kk, jj, :] + sqrt_beta_y[kk, jj, :]
        # terminal conditions
        mat_1 = Gf @ Phi_xx[N, jj] @ Ejj 
        mat_2 = Gf @ Phi_xy[N, jj] @ Fjj 
        sqrt_beta_x_f[jj, :] = np.sqrt(np.sum(mat_1**2, axis=1))
        sqrt_beta_y_f[jj, :] = np.sqrt(np.sum(mat_2**2, axis=1))
        beta_f[jj, :] = sqrt_beta_x_f[jj, :] ** 2 + sqrt_beta_y_f[jj, :] ** 2
        sqrt_beta_f[jj, :] = sqrt_beta_x_f[jj, :] + sqrt_beta_y_f[jj, :]
    
    return beta, beta_f, sqrt_beta, sqrt_beta_f, \
            sqrt_beta_x, sqrt_beta_x_f, sqrt_beta_y, sqrt_beta_y_f

def generate_tightening(N, ni, ni_f, Phi_xx, Phi_ux, Phi_xy, Phi_uy, E, F, G, Gf):
    Phi_xx_np = np64_contig(Phi_xx)
    Phi_ux_np = np64_contig(Phi_ux)
    Phi_xy_np = np64_contig(Phi_xy)
    Phi_uy_np = np64_contig(Phi_uy)
    
    return scan_tightening(N, ni, ni_f, Phi_xx_np, Phi_ux_np, Phi_xy_np, Phi_uy_np, E, F, G, Gf)

def warmup():
    """
    compile the parallel function
    """
    print("start warming up Numba for parallel computing")
    N = 2
    nx = 1 
    nu = 1
    ny = 1
    ni = 1
    ni_f = 1
    
    Q_reg = np.eye(nx)
    R_reg = np.eye(nu)
    Q_reg_f = np.eye(nx)
    Q_reg_obs = np.eye(nx)
    R_reg_obs = np.eye(ny)
    Q_reg_f_obs = np.eye(nx)
    eta = np.zeros((N, N, ni))
    eta_f = np.zeros((N + 1, ni_f))
    A = [np.eye(nx)] * N
    B = [np.eye(nu)] * N
    C = [np.eye(ny)] * N
    E = [np.eye(nx)] * N
    F = [np.eye(ny)] * N
    G = np.eye(ni)
    Gf = np.eye(ni_f)
    I = np.ascontiguousarray(np.eye(nx))

    generate_state_Phi(N, nx, nu, A, B, Q_reg, R_reg, Q_reg_f, 
                                                  eta, eta_f, G, Gf)
    generate_observer_Phi(N, nx, ny, A, C, E, F, Q_reg_obs, R_reg_obs, Q_reg_f_obs)
    
    Phi_xx = np.zeros((N+1, N+1, nx, nx))
    Phi_ux = np.zeros((N+1, N+1, nu, nx))
    Phi_xy = np.zeros((N+1, N+1, nx, ny))
    Phi_uy = np.zeros((N+1, N+1, nu, ny))
    generate_tightening(N, ni, ni_f, Phi_xx, Phi_ux, Phi_xy, Phi_uy, E, F, G, Gf)

    print("warmup finished")
    print("----------------------")
