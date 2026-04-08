import casadi as ca
import numpy as np
from scipy.linalg import block_diag
from dyn.LTI import LTI
from dyn.LTV import LTV
from dyn.LTI_OF import LTI_OF
from dyn.LTV_OF import LTV_OF

from solver.ocp import OCP
from solver.ocp_of import OCP_OF
from solver.qp import QP
from solver.controller_optimizer import optimize_tube
from dyn.LTI import LTI
from dyn.LTV import LTV
from util.SLS import SLS

from matplotlib import pyplot as plt
from prettytable import PrettyTable, NONE, HEADER

import time

from util.parallel_riccati import generate_tightening


class fast_OF_SLS(OCP_OF):

    def __init__(self, N, Q, R, m, Qf, Q_reg=None, R_reg=None, Q_reg_f=None, 
                 parallel=False, obstacles=[]):
        super().__init__(N, Q, R, m, Qf, Q_reg, R_reg, Q_reg_f)

        # parameter solver
        self.epsilon_backoff = 1e-10
        self.MAX_ITER = 25 
        self.CONV_EPS = 1e-3
        self.current_iteration = {}  # structure that contains current iteration data
        self.convergence_data = {}  # timings, number of iterations, convergence, etc.
        self.save_it_data = True
        self.it_data = {}  # structure that contains iteration data
        self.verbose = True
        self.parallel = parallel

        # dynamics matrices
        self.initialize_list_dynamics()

        # forward solver objects
        self.solver_forward = None
        self.nominal_ubg = None  # nominal upper bound on the inequality constraints (without backoff)
        self.nominal_lbg = None  # nominal lower bound on the inequality constraints (without backoff)
        self.initialize_solver_forward(obstacles)

        self.initialize_solver()
    
    def solve(self, x0, obstacles=[], primal_pos=[]):
        self.initialize_backoff()
        if self.verbose:
            table = self.printHeader()
        else:
            table = None

        for i in range(self.MAX_ITER):
            # print(obstacles)
            step_result = self._step(x0, i, table, obstacles, primal_pos)
            if step_result is False:
                # infeasible forward solve
                return self._finish_failure(i, infeasible=True)
            if step_result is True:
                # converged
                return self._finish_success(i)
        # ran out of iterations
        return self._finish_failure(self.MAX_ITER - 1)
    
    def _step(self, x0, i, table, obstacles=[], primal_pos=[]):
        """Perform one SLS iteration. Returns True if converged,
           False if infeasible, None otherwise."""
        if not self.forward_solve(x0, obstacles):
            return False

        self.evaluate_dual_eta()

        self.backward_solve()

        self.update_tightening(obstacles, primal_pos)

        if self.check_convergence_socp():
            self.current_iteration['success'] = True
            self.convergence_data['iterations'] = i
            return True

        if self.verbose and table is not None:
            self.printLine(i, table)

        if self.save_it_data:
            self.it_data[i] = self.current_iteration.copy()

        return None
    
    def _finish_success(self, i):
        if self.verbose:
            print(f"Fast-SLS: Converged in {i} iterations")
        return self.post_processing_solution()

    def _finish_failure(self, i, infeasible=False):
        if infeasible:
            self.current_iteration['success'] = False
        else:
            self.current_iteration['success'] = True
        self.convergence_data['iterations'] = i
        if self.verbose:
            msg = "infeasible forward solve" if infeasible else f"did not converge in {i} iters"
            print(f"Fast‑SLS: {msg}")
        sol = self.post_processing_solution()
        self.reset_solver_to_zeros()
        return sol
    
    def initialize_solver(self):
        N = self.N
        ni = self.m.ni
        ni_f = self.m.ni_f

        self.current_iteration['cost_nominal'] = np.nan
        self.current_iteration['cost_tube'] = np.nan
        self.current_iteration['cost'] = np.nan

        self.current_iteration['primal_vec'] = np.nan
        self.current_iteration['dual_vec'] = np.nan

        self.current_iteration['eta'] = np.full((N, N, ni), np.nan)
        self.current_iteration['eta_f'] = np.full((N + 1, ni_f), np.nan)

    def reset_solver_to_zeros(self):
        """
        This method resets the solver of the fast-SLS algorithm such that it can be used with new initial conditions. In particular, it
        resets the primal and dual solutions, the backoff, and the solver matrices.
        :return:
        """
        self.current_iteration['primal_vec'] = np.nan
        self.current_iteration['primal_x'] = np.nan
        self.current_iteration['primal_u'] = np.nan
        self.current_iteration['previous_primal_vec'] = np.nan

        self.current_iteration['dual_vec'] = np.nan
        self.current_iteration['dual_mu'] = np.nan
        self.current_iteration['dual_mu_f'] = np.nan
        self.current_iteration['previous_dual_vec'] = np.nan

        self.solver_forward.reset_ubg()
        self.solver_forward.reset_q_cost_lin()
        self.solver_forward.reset_q_cost_quad()
        self.solver_forward.lbg = self.nominal_lbg

        self.initialize_backoff()

        self.current_iteration['M'] = 0.

        self.current_iteration['eta'] = np.zeros((self.N, self.N, self.m.ni))
        self.current_iteration['eta_f'] = np.zeros((self.N + 1, self.m.ni_f))
        self.current_iteration['previous_eta'] = np.zeros((self.N, self.N, self.m.ni))
        self.current_iteration['previous_eta_f'] = np.zeros((self.N + 1, self.m.ni_f))

        self.current_iteration['cost_nominal'] = np.nan
        self.current_iteration['cost_tube'] = np.nan
        self.current_iteration['cost'] = np.nan

    def reset_solver_to_warm_start(self):
        raise NotImplementedError
    
    def initialize_solver_forward(self, obstacles=[]):
        """
        This method initializes the solver for the forward pass of the fast-SLS algorithm.
        In particular, it creates a QP solver object that contains the structure of the optimization problem.
        """
        m = self.m
        self.solver_forward = QP(self.N, self.Q, self.R, m, self.Qf, obstacles)
        self.nominal_lbg = self.solver_forward.lbg
        return self.solver_forward
    
    def forward_solve(self, x0, obstacles=[]):
        """
        This method solves the forward pass of the fast-SLS algorithm. In particular, it solves a linear trajectory
        optimization problem. For each iteration of the fast-SLS algorithm, it solves the problem with different
        backoff.
        :param x0: initial conditions
        :return:
        """
        # todo: need to update the matrices A and B in the solver_forward object
        solution_forward = self.solver_forward.solve(x0, obstacles)

        if solution_forward['success'] is False:
            if self.verbose:
                print('Fast-SLS: Infeasible forward solution. Try with another initial condition.')

            return solution_forward['success']

        # extract the solution
        # primal solution
        primal_y = solution_forward['primal_vec']
        primal_x = solution_forward['primal_x']
        primal_u = solution_forward['primal_u']

        # dual solution
        dual = solution_forward['dual_vec']
        # extract the dual associated to the terminal constraints
        mu_f = solution_forward['dual_mu_f']
        mu = solution_forward['dual_mu']

        # save previous primal and dual solutions
        self.current_iteration['previous_primal_vec'] = self.current_iteration['primal_vec']
        self.current_iteration['previous_dual_vec'] = self.current_iteration['dual_vec']

        # update the current iteration data
        self.current_iteration['primal_vec'] = primal_y
        self.current_iteration['primal_x'] = primal_x
        self.current_iteration['primal_u'] = primal_u

        self.current_iteration['dual_vec'] = dual
        self.current_iteration['dual_mu'] = mu  # transpose to have the same index ordering as other time series
        self.current_iteration['dual_mu_f'] = mu_f  # transpose to have the same index ordering as other time series

        self.current_iteration['cost_nominal'] = solution_forward['cost']
        return solution_forward['success']

    def initialize_backoff(self):
        """
        Initialize the backoff to some epsilon values for the first iteration
        :return:
        """
        # initialize the backoff
        self.current_iteration['beta'] = np.zeros((self.N, self.N, self.m.ni)) + self.epsilon_backoff
        self.current_iteration['beta_f'] = np.zeros((self.N + 1, self.m.ni_f)) + self.epsilon_backoff

        self.current_iteration['beta_x'] = np.zeros((self.N, self.N, self.m.ni)) + self.epsilon_backoff
        self.current_iteration['beta_x_f'] = np.zeros((self.N + 1, self.m.ni_f)) + self.epsilon_backoff

        self.current_iteration['beta_y'] = np.zeros((self.N, self.N, self.m.ni)) + self.epsilon_backoff
        self.current_iteration['beta_y_f'] = np.zeros((self.N + 1, self.m.ni_f)) + self.epsilon_backoff

        self.current_iteration['sqrt_beta'] = np.zeros((self.N, self.N, self.m.ni)) + np.sqrt(self.epsilon_backoff)
        self.current_iteration['sqrt_beta_f'] = np.zeros((self.N + 1, self.m.ni_f)) + np.sqrt(self.epsilon_backoff)

        self.current_iteration['backoff'] = np.nansum(self.current_iteration["sqrt_beta"], axis=1)
        self.current_iteration['backoff_f'] = np.sum(self.current_iteration["sqrt_beta_f"], axis=0).T

        self.current_iteration['backoff_x'] = np.zeros((self.N + 1, self.m.nx))
        self.current_iteration['backoff_u'] = np.zeros((self.N, self.m.nu))

    def update_dynamics_list(self, new_list_A, new_list_B, new_list_E=None, new_list_g=None, c_offset_list=None, 
                             obstacles=[], primal_pos=[], loose_terminal=False):
        """
        Update the linear dynamics.
        :return:
        """
        # update the dynamics
        self.A_list = new_list_A
        self.B_list = new_list_B

        if new_list_E is not None:
            self.E_list = new_list_E

        if new_list_g is not None:
            # update the g_list for all elements of the terminal constraints
            self.g_list = new_list_g

        # update the solver_forward object with the new dynamics
        self.solver_forward.update_dynamics(self.A_list, self.B_list, self.E_list, self.g_list, obstacles, primal_pos, loose_terminal)
        # todo: make sure the g_list is well taken care off.

        if c_offset_list is not None:
            self.c_offset_list = c_offset_list
            self.solver_forward.offset_constraints(np.hstack(c_offset_list), obstacles)

    def evaluate_dual_eta(self):
        """
        This method computes the dual variables eta_kj.
        :return:
        """
        N = self.N

        # initialization the eta matrix with Nan values
        eta = np.full((N, N, self.m.ni), np.nan)
        eta_f = np.full((N + 1, self.m.ni_f), np.nan)

        beta = self.current_iteration['beta']
        beta_f = self.current_iteration['beta_f']

        # if value of beta is too small, set it to self.epsilon_backoff
        beta = np.maximum(beta, self.epsilon_backoff)
        beta_f = np.maximum(beta_f, self.epsilon_backoff)

        for jj in range(N):
            for kk in range(jj, N):
                eta[kk, jj, :] = self.current_iteration['dual_mu'][:, kk] / np.sqrt(beta[kk, jj, :]) / 2.

        for jj in range(N + 1):
            eta_f[jj, :] = self.current_iteration['dual_mu_f'] / np.sqrt(beta_f[jj, :]) / 2.

        # update the current iteration data for eta_kj
        self.current_iteration['previous_eta'] = self.current_iteration['eta']
        self.current_iteration['previous_eta_f'] = self.current_iteration['eta_f']

        self.current_iteration['eta'] = eta
        self.current_iteration['eta_f'] = eta_f

    def check_convergence_socp(self):
        """
        This method checks the convergence of the fast SLS algorithm.
        :return:
        """
        # print('Convergence checked on nominal trajectory')
        delta_primal = np.max(
            self.current_iteration['primal_vec'] - self.current_iteration['previous_primal_vec'])

        # replace the NaN values by zeros
        delta_dual = np.max(np.nan_to_num(
            self.current_iteration['eta'] - self.current_iteration['previous_eta']))
        
        delta_backoff = np.max(np.fabs(np.nan_to_num(self.current_iteration['backoff'] - self.current_iteration['previous_backoff'])))
        
        return delta_primal <= self.CONV_EPS and delta_backoff <= self.CONV_EPS

    def post_processing_solution(self):
        if self.current_iteration['success']:
            Phi_xx_mat = self.current_iteration['Phi_xx_mat']
            Phi_ux_mat = self.current_iteration['Phi_ux_mat']
            Phi_xy_mat = self.current_iteration['Phi_xy_mat']
            Phi_uy_mat = self.current_iteration['Phi_uy_mat']

            Phi_xx = SLS.convert_matrix_to_tensor(Phi_xx_mat, self.N + 1, self.m.nx, self.m.nx)
            Phi_ux = SLS.convert_matrix_to_tensor(Phi_ux_mat, self.N + 1, self.m.nu, self.m.nx)
            Phi_xy = SLS.convert_matrix_to_tensor(Phi_xy_mat, self.N + 1, self.m.nx, self.m.ny)
            Phi_uy = SLS.convert_matrix_to_tensor(Phi_uy_mat, self.N + 1, self.m.nu, self.m.ny)

            self.current_iteration['Phi_xx'] = Phi_xx
            self.current_iteration['Phi_ux'] = Phi_ux
            self.current_iteration['Phi_xy'] = Phi_xy
            self.current_iteration['Phi_uy'] = Phi_uy

        solution = self.current_iteration.copy()
        solution.update(self.convergence_data.copy())
        solution['it_data'] = self.it_data.copy()
        solution['cost'] = self.current_iteration['cost_tube'] + self.current_iteration['cost_nominal']

        return solution
    
    def backward_solve(self):
        m = self.m
        N = self.N
        nx = m.nx
        nu = m.nu
        ny = m.ny

        G_f = m.Gf
        G = m.G

        A = self.A_list
        B = self.B_list
        C = self.C_list
        E = self.E_list
        F = self.F_list

        Phi_xx_cvxpy, Phi_ux_cvxpy, Phi_xy_cvxpy, Phi_uy_cvxpy, cost = \
            optimize_tube(m, N, A, B, C, E, F, 
                          self.Q_reg, self.R_reg, self.Q_reg_f, 
                          self.current_iteration["eta"], self.current_iteration["eta_f"])
        Phi_xx_mat = SLS.convert_tensor_to_matrix(Phi_xx_cvxpy)
        Phi_ux_mat =  SLS.convert_tensor_to_matrix(Phi_ux_cvxpy)
        Phi_xy_mat = SLS.convert_tensor_to_matrix(Phi_xy_cvxpy)
        Phi_uy_mat =  SLS.convert_tensor_to_matrix(Phi_uy_cvxpy)
        self.current_iteration['Phi_xx_mat'] = Phi_xx_mat
        self.current_iteration['Phi_ux_mat'] = Phi_ux_mat
        self.current_iteration['Phi_xy_mat'] = Phi_xy_mat
        self.current_iteration['Phi_uy_mat'] = Phi_uy_mat
        
        # self.current_iteration['cost_tube'] = cost
        Phi_mat = np.hstack([np.vstack([Phi_xx_mat, Phi_ux_mat]), np.vstack([Phi_xy_mat, Phi_uy_mat])])

        Q_reg = np.sqrt(self.Q_reg)
        R_reg = np.sqrt(self.R_reg)
        Q_reg_f = np.sqrt(self.Q_reg_f)
        error_bounds = block_diag(*self.E_list)
        error_bounds = block_diag(error_bounds, *self.F_list)

        Q_reg_blk = block_diag(np.kron(np.eye(N), Q_reg), Q_reg_f)
        R_reg_blk = block_diag(np.kron(np.eye(N + 1), R_reg))

        cost = np.linalg.norm(block_diag(Q_reg_blk, R_reg_blk) @ Phi_mat @ error_bounds, ord='fro')**2
        self.current_iteration['cost_tube'] = cost
        self.current_iteration['cost'] = self.current_iteration['cost_tube'] + self.current_iteration['cost_nominal']
    
    def update_tightening(self, obstacles=[], primal_pos=[]):
        """
        This function calculate the backff parameters for the robust constraint
        """
        N = self.N
        nx = self.m.nx
        nu = self.m.nu
        ny = self.m.ny

        nw = self.m.nw
        ni = self.m.ni
        ni_f = self.m.ni_f

        G = self.m.G
        Gf = self.m.Gf

        A = self.A_list
        B = self.B_list
        E = self.E_list
        C = self.C_list
        F = self.F_list

        Phi_xx_mat = self.current_iteration['Phi_xx_mat']
        Phi_ux_mat = self.current_iteration['Phi_ux_mat']
        Phi_xy_mat = self.current_iteration['Phi_xy_mat']
        Phi_uy_mat = self.current_iteration['Phi_uy_mat']

        Phi_xx = SLS.convert_matrix_to_tensor(Phi_xx_mat, N + 1, nx, nx)
        Phi_ux = SLS.convert_matrix_to_tensor(Phi_ux_mat, N + 1, nu, nx)
        Phi_xy = SLS.convert_matrix_to_tensor(Phi_xy_mat, N + 1, nx, ny)
        Phi_uy = SLS.convert_matrix_to_tensor(Phi_uy_mat, N + 1, nu, ny)

        self.current_iteration['Phi_xx'] = Phi_xx
        self.current_iteration['Phi_ux'] = Phi_ux
        self.current_iteration['Phi_xy'] = Phi_xy
        self.current_iteration['Phi_uy'] = Phi_uy

        # initialize the backoff
        if not self.parallel:
            beta = np.full((N, N, ni), np.nan)  # backoff
            beta_f = np.full((N + 1, ni_f), np.nan)

            sqrt_beta = np.full((N, N, ni), np.nan)
            sqrt_beta_f = np.full((N + 1, ni_f), np.nan)

            sqrt_beta_x = np.full((N, N, ni), np.nan)
            sqrt_beta_x_f = np.full((N + 1, ni_f), np.nan)
            sqrt_beta_y = np.full((N, N, ni), np.nan)
            sqrt_beta_y_f = np.full((N + 1, ni_f), np.nan)

            # calculate the backoff
            # column-wise
            # TODO: also parallelize this
            for jj in range(N + 1):
                # row-wise
                for kk in range(jj, N):
                    # evaluate the 2-norm of each component
                    mat_1 = G @ np.vstack([Phi_xx[kk, jj], Phi_ux[kk, jj]]) @ E[jj]
                    mat_2 = G @ np.vstack([Phi_xy[kk, jj], Phi_uy[kk, jj]]) @ F[jj]
                    sqrt_beta_x[kk, jj, :] = np.linalg.norm(mat_1, axis=1)
                    sqrt_beta_y[kk, jj, :] = np.linalg.norm(mat_2, axis=1)
                    beta[kk, jj, :] = sqrt_beta_x[kk, jj, :] ** 2 + sqrt_beta_y[kk, jj, :] ** 2
                    sqrt_beta[kk, jj, :] = sqrt_beta_x[kk, jj, :] + sqrt_beta_y[kk, jj, :]
                # terminal conditions
                mat_1 = Gf @ Phi_xx[N, jj] @ E[jj] 
                mat_2 = Gf @ Phi_xy[N, jj] @ F[jj] 
                sqrt_beta_x_f[jj, :] = np.linalg.norm(mat_1, axis=1)
                sqrt_beta_y_f[jj, :] = np.linalg.norm(mat_2, axis=1)
                beta_f[jj, :] = sqrt_beta_x_f[jj, :] ** 2 + sqrt_beta_y_f[jj, :] ** 2
                sqrt_beta_f[jj, :] = sqrt_beta_x_f[jj, :] + sqrt_beta_y_f[jj, :]
        else:
            beta, beta_f, sqrt_beta, sqrt_beta_f, sqrt_beta_x, sqrt_beta_x_f, sqrt_beta_y, sqrt_beta_y_f = \
                    generate_tightening(N, ni, ni_f, Phi_xx, Phi_ux, Phi_xy, Phi_uy, E, F, G, Gf)
        # update backoff of current iteration
        self.current_iteration['beta'] = beta
        self.current_iteration['beta_f'] = beta_f
        
        self.current_iteration['beta_x'] = sqrt_beta_x ** 2
        self.current_iteration['beta_x_f'] = sqrt_beta_x_f ** 2
        self.current_iteration['beta_y'] = sqrt_beta_y ** 2
        self.current_iteration['beta_y_f'] = sqrt_beta_y_f ** 2

        # evaluate new backoff as the sum of the contribution of each disturbance
        backoff = np.nansum(sqrt_beta, axis=1)
        backoff_f = np.sum(sqrt_beta_f, axis=0).T  # transpose to recover convention for time propagation
        
        self.current_iteration['previous_backoff'] = self.current_iteration['backoff']

        self.current_iteration['backoff'] = backoff
        self.current_iteration['backoff_f'] = backoff_f

        self.current_iteration['backoff_x'] = np.vstack((backoff[:, :nx], backoff_f[:nx]))
        self.current_iteration['backoff_u'] = backoff[:, nx:nx + nu]

        backoff_process = np.nansum(sqrt_beta_x, axis=1)
        backoff_process_f = np.sum(sqrt_beta_x_f, axis=0).T
        backoff_measurement = np.nansum(sqrt_beta_y, axis=1)
        backoff_measurement_f = np.sum(sqrt_beta_y_f, axis=0).T

        self.current_iteration['backoff_process'] = backoff_process
        self.current_iteration['backoff_process_f'] = backoff_process_f
        self.current_iteration['backoff_measurement'] = backoff_measurement
        self.current_iteration['backoff_measurement_f'] = backoff_measurement_f

        g = self.g_list
        gf = np.squeeze(self.g_list[-1])
        absolute_backoff_table = np.squeeze(g[:-1]) - backoff
        
        collision_avoidance_ubg = []
        for obstacle in obstacles:
            center = obstacle[0]
            rad = obstacle[1]
            for t in range(1, N):
                over_approx = np.sqrt(backoff[t,0]**2 + backoff[t,1]**2)
                collision_avoidance_ubg.append(np.linalg.norm(primal_pos[:,t] - center) - rad - over_approx) 
            over_approx = np.sqrt(backoff_f[0]**2 + backoff_f[1]**2)
            collision_avoidance_ubg.append(np.linalg.norm(primal_pos[:,N] - center) - rad - over_approx)
        collision_avoidance_ubg = np.array(collision_avoidance_ubg)

        c_offset_mat = np.hstack(self.c_offset_list)
        new_ubg_table = np.vstack([-c_offset_mat, absolute_backoff_table.T])
        new_ubg_without_terminal = np.reshape(new_ubg_table, (N * (ni + nx)), order='F')
        new_ubg = np.concatenate([new_ubg_without_terminal, gf - backoff_f, collision_avoidance_ubg])
        # todo: we should also update the equality constraints with c_offset_list
        self.solver_forward.update_ubg(new_ubg)

    def update_observer_list(self, new_list_C=None, new_list_F=None):

        if new_list_C is not None:
            self.C_list = new_list_C
        if new_list_F is not None:
            self.F_list = new_list_F
    
    def initialize_tube_cost_fun(self,
            A_fun,  # list of length N of casadi.Function, each: y -> A_k(z,v) ∈ ℝⁿˣⁿ
            B_fun,  # list of length N of casadi.Function, each: y -> B_k(z,v) ∈ ℝⁿˣᵘ
            E_fun,  # list of length N+1 of casadi.Function, each: y -> E_j(z) ∈ ℝⁿˣʷ
            F_fun,
    ):
        """
        This method creates a CasADi function that computes the tightening of the constraints based on the current state and control inputs.
        :param A_fun:
        :param B_fun:
        :param E_fun:
        :return:
        """

        # — create casadi constants —
        G = ca.DM(self.m.G)
        Gf = ca.DM(self.m.Gf)
        nu = self.m.nu
        nw = self.m.nw
        nx = self.m.nx
        ny = self.m.ny
        nv = self.m.nv
        ni = self.m.ni
        N = self.N

        # — symbolic inputs —
        # Z = ca.SX.sym('Z', nx, (N + 1))
        # V = ca.SX.sym('V', nu,  N)
        vec = ca.SX.sym('vec', nx*(N+1) + nu*N + self.m.ni_f)
        Z = [vec[i*(nx+nu):i*(nx+nu)+nx] for i in range(N+1)]
        V = [vec[i*(nx+nu)+nx:(i+1)*(nx+nu)] for i in range(N)]

        # disturbance feedback parts
        # TODO: this can be optimized (only half is needed)
        Phi_xx = [
            [ca.SX.sym(f"Phi_xx_{k}_{j}", nx, nw) for j in range(N + 1)]
            for k in range(N+1)
        ]
        Phi_ux = [
            [ca.SX.sym(f"Phi_ux_{k}_{j}", nu, nw) for j in range(N + 1)]
            for k in range(N+1)
        ]
        Phi_xy = [
            [ca.SX.sym(f"Phi_xy_{k}_{j}", nx, nv) for j in range(N + 1)]
            for k in range(N+1)
        ]
        Phi_uy = [
            [ca.SX.sym(f"Phi_uy_{k}_{j}", nu, nv) for j in range(N + 1)]
            for k in range(N+1)
        ]
    
        E = [E_fun(Z[j], V[j]) for j in range(N)] + [E_fun(Z[N], ca.DM.zeros(nu, 1))]
        F = [F_fun(Z[j]) for j in range(N+1)]

        # compute tube cost
        cost = 0
        Q_reg = ca.DM(np.sqrt(self.Q_reg))
        R_reg = ca.DM(np.sqrt(self.R_reg))
        Q_reg_f = ca.DM(np.sqrt(self.Q_reg_f))
        for jj in range(N):
            for kk in range(jj, N):
                cost += ca.sumsqr(Q_reg @ Phi_xx[kk][jj] @ E[jj]) + ca.sumsqr(R_reg @ Phi_ux[kk][jj] @ E[jj])
                cost += ca.sumsqr(Q_reg @ Phi_xy[kk][jj] @ F[jj]) + ca.sumsqr(R_reg @ Phi_uy[kk][jj] @ F[jj])

        for jj in range(N + 1):
            cost += ca.sumsqr(Q_reg_f @ Phi_xx[N][jj] @ E[jj]) + ca.sumsqr(Q_reg_f @ Phi_xy[N][jj] @ F[jj])

        # assemble inputs & outputs
        inputs = [vec] + sum(Phi_xx, []) + sum(Phi_ux, []) + sum(Phi_xy, []) + sum(Phi_uy, [])
        outputs = [cost]

        return ca.Function('tighten', inputs, outputs, {'jit': False})

    def update_linear_cost(self, q_cost_lin):
        """
        This method updates the linear cost of the forward solver.
        :param q_cost_lin:
        :return:
        """
        self.solver_forward.update_q_cost_lin(q_cost_lin)

    def add_linear_cost(self, q_cost_lin):
        """
        This method adds a linear cost to the forward solver.
        :param q_cost_lin:
        :return:
        """
        self.solver_forward.add_q_cost_lin(q_cost_lin)

    def update_quadratic_cost(self, q_cost_quad):
        self.solver_forward.update_q_cost_quad(q_cost_quad)

    def printLine(self, i, table):
        """
        This method prints the current iteration data in a formatted table.
        """
        fixed_width = 10
        primal = np.max(self.current_iteration['primal_vec'])
        dual = np.max(np.nan_to_num(self.current_iteration['eta']))
        cost = self.current_iteration['cost']
        iteration_str = f"{i:>{fixed_width}}"
        primal_val = f"{float(primal):>{fixed_width}.2e}"
        dual_val = f"{float(dual):>{fixed_width}.2e}"
        cost_val = f"{float(cost):>{fixed_width}.2e}"
        # todo add absolute primal and dual values

        table.add_row([iteration_str, primal_val, dual_val, cost_val])
        print(table.get_string(start=len(table._rows) - 1, end=len(table._rows), header=False))

    @staticmethod
    def printHeader():
        fixed_width = 10
        # Format headers to have fixed width and right alignment
        headers = ["it (fast-SLS)", "primal", "dual", "cost"]
        formatted_headers = [f"{h:>{fixed_width}}" for h in headers]
        table = PrettyTable()
        table.field_names = formatted_headers
        table.hrules = HEADER  # Horizontal line after header
        table.border = True

        # Align columns
        table.align["it"] = "right"
        table.align["primal"] = "right"
        table.align["dual"] = "right"
        table.align["cost"] = "right"

        # Set a fixed width for all columns
        fixed_width = 10
        # Set fixed widths for each column individually
        table.max_width["it"] = fixed_width
        table.max_width["primal"] = fixed_width
        table.max_width["dual"] = fixed_width
        table.max_width["cost"] = fixed_width

        print(table.get_string(end=0))
        table.hrules = NONE

        return table

    
