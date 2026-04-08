import casadi as ca
import numpy as np
from scipy.linalg import block_diag
from dyn.LTI import LTI
from dyn.LTV import LTV
from dyn.LTI_OF import LTI_OF
from dyn.LTV_OF import LTV_OF

from solver.qp import QP
from dyn.LTI import LTI
from dyn.LTV import LTV
from util.SLS import SLS

from matplotlib import pyplot as plt

from solver.fast_OF_SLS_SDP import fast_OF_SLS as fast_OF_SLS_cvxpy
from solver.ocp_of import OCP_OF
from solver.nlp import NLP
from solver.controller_optimizer import optimize_tube

from prettytable import PrettyTable, NONE, HEADER

import time

class SCP_OF_SLS(OCP_OF):
    """
    This class is an implementation of iSLS
    """

    def __init__(self, N, Q, R, m, Qf, Q_reg=None, R_reg=None, Q_reg_f=None, 
                 H_reg_coef=2., parallel=False, obstacles=[], init_guess=None, **kwargs):
        super().__init__(N, Q, R, m, Qf, Q_reg, R_reg, Q_reg_f)
        self.H_reg_coef = H_reg_coef

        self.c_offset_dynamics_fun = None
        self.H_mat = None
        self.epsilon_convergence = 1e-3 
        self.current_iteration_scp = {}  # structure that contains current iteration data
        self.convergence_data_scp = {}  # timings, number of iterations, convergence, etc.
        self.it_data = {}  # structure that contains iteration data
        self.save_it_data = kwargs.get("save_it_data", True)
        self.verbose = kwargs.get("verbose", True)

        # This flag indicates whether the linearization error is used. Only with an overbound of the linearization error, we can guarantee robust constraint satisfaction for the uncertain nonlinear dynamics.
        self.linearization_error = kwargs.get("linearization_error", False)

        self.nominal_trajectory_solver = None
        self.solver_nominal_params = {}
        self.initialize_nominal_trajectory_solver(obstacles, init_guess)

        self.MAX_ITER_SCP = 20 
        self.A_fun = None
        self.B_fun = None
        self.E_fun = None
        self.C_fun = None
        self.F_fun = None
        self.initialize_jacobian_Function()
        
        self.parallel = parallel
        self.fast_SLS_solver = None
        self.initialize_fast_SLS_solver(obstacles)
        self.fast_SLS_solver.verbose = True
        self.fast_SLS_solver.save_it_data = True
        
        self.state_dependent = kwargs.get("state_dependent", True)
        if self.state_dependent:
            self.initialize_tube_cost_quadratization()

        self.initialization_file = kwargs.get("initialization_file", None)

    def solve(self, x0, xG=None, obstacles=[]):
        if self.verbose:
            table = self.printHeader()
        if xG is None:
            xG = np.zeros(self.m.nx)

        # nominal trajectory initialization
        if not self.solve_nominal_trajectory(x0, xG):
            return {'success': False}
        # --------------------
        if len(xG.shape) == 1:
            self.goal_offset = np.concatenate([np.tile(np.concatenate([xG, np.zeros((self.m.nu))]), self.N), xG, np.zeros((self.m.ni_f))])
        elif len(xG.shape) == 2:
            self.goal_offset = []
            for i in range(xG.shape[0]-1):
                self.goal_offset = np.concatenate([self.goal_offset, xG[i,:], np.zeros((self.m.nu))])
            self.goal_offset = np.concatenate([self.goal_offset, xG[-1,:], np.zeros((self.m.ni_f))])
        else:
            raise NotImplementedError("Unsupported goal shape")
        self.goal_offset = self.goal_offset.reshape((-1,1))
        # --------------------

        for ii in range(self.MAX_ITER_SCP):
            self.update_jacobian(obstacles, ii)

            # solve fast-SLS algorithm
            if not self.socp_step(obstacles):
                break

            if ii == 0:
                self.current_iteration_scp["initial_backoff_x"] = self.current_iteration_scp["backoff_x"]

            if self.verbose:
                self.printLine(ii, table)

            if self.check_convergence_scp():
                self.current_iteration_scp['success'] = True
                self.current_iteration_scp['iterations'] = ii
                if self.verbose:
                    print('SCP-SLS: Solution found! Converged in {} iterations'.format(ii))
                solution = self.post_processing_solution()
                return solution #self.current_iteration_scp

            if self.save_it_data:
                self.it_data[ii] = self.current_iteration_scp.copy()

        self.current_iteration_scp['success'] = False
        if self.verbose:
            print('SCP did not converge in {} iterations'.format(ii))
        solution = self.post_processing_solution()
        return solution
    
    def initialize_nominal_trajectory_solver(self, obstacles=[], init_guess=None):
        """
        This method initializes the nominal trajectory solver
        :return:
        """
        self.nominal_trajectory_solver = NLP(self.N, self.Q, self.R, self.m, self.Qf, obstacles, init_guess)

    def solve_nominal_trajectory(self, x0, xG):
        # --------------------
        if self.initialization_file is not None:
            sol = np.load(self.initialization_file) 
            self.current_iteration_scp['primal_x'] = sol['initial_traj']
            self.current_iteration_scp['primal_u'] = sol['initial_input']
            self.current_iteration_scp['primal_vec'] = sol['initial_vec']
            self.current_iteration_scp['dual_vec'] = sol['initial_dual_vec']
            self.current_iteration_scp['cost_nominal'] = sol['initial_cost']
            
            self.current_iteration_scp['initial_x'] = sol['initial_traj']
            self.current_iteration_scp['initial_u'] = sol['initial_input']

            # self.initialize_M()
            return True
        # --------------------

        sol = self.nominal_trajectory_solver.solve(x0, xG)
        if sol['success']:
            self.current_iteration_scp['primal_x'] = np.copy(sol['primal_x'])
            self.current_iteration_scp['primal_u'] = np.copy(sol['primal_u'])
            self.current_iteration_scp['primal_vec'] = np.concatenate([sol['primal_vec'], np.zeros((self.m.ni_f, 1))])
            self.current_iteration_scp['dual_vec'] = np.copy(sol['dual_vec'])
            self.current_iteration_scp['cost_nominal'] = sol['cost']
            
            self.current_iteration_scp['initial_x'] = np.copy(sol['primal_x'])
            self.current_iteration_scp['initial_u'] = np.copy(sol['primal_u'])

            # self.initialize_M()

        if self.verbose:
            if sol['success']:
                print('SCP-SLS: Initial guess nominal trajectory converged!')
            else:
                print('SCP-SLS: Initial guess nominal trajectory did not converge!')

        return sol['success']
    
    def initialize_M(self, A_list, B_list, C_list, E_list, F_list):
        m = self.m
        N = self.N
        nx = m.nx
        nu = m.nu
        nw = m.nw
        ny = m.ny
        nv = m.nv

        G_f = m.Gf
        G = m.G

        Q_reg = self.Q_reg
        R_reg = self.R_reg
        Q_reg_f = self.Q_reg_f

        Phi_xx_cvxpy, Phi_ux_cvxpy, Phi_xy_cvxpy, Phi_uy_cvxpy, cost = \
            optimize_tube(m, N, A_list, B_list, C_list, E_list, F_list, 
                          Q_reg, R_reg, Q_reg_f, 
                           np.zeros((N, N, m.ni)), np.zeros((N + 1, m.ni_f)))

        self.current_iteration_scp['Phi_xx'] = Phi_xx_cvxpy
        self.current_iteration_scp['Phi_ux'] = Phi_ux_cvxpy
        self.current_iteration_scp['Phi_xy'] = Phi_xy_cvxpy
        self.current_iteration_scp['Phi_uy'] = Phi_uy_cvxpy
    
    def initialize_jacobian_Function(self):
        """
        This method initializes the Jacobian functions.
        :return:
        """
        # TODO: check if the implementation is correct
        x = ca.MX.sym('x', self.m.nx)
        u = ca.MX.sym('u', self.m.nu)
        w = ca.MX.sym('w', self.m.nw)
        v = ca.MX.sym('v', self.m.nv)
        xp = ca.MX.sym('xp', self.m.nx)

        A = ca.jacobian(self.m.ddyn(x, u, w, self.m.dt), x)
        B = ca.jacobian(self.m.ddyn(x, u, w, self.m.dt), u)
        E = ca.jacobian(self.m.ddyn(x, u, w, self.m.dt), w)
        C = ca.jacobian(self.m.measurement(x, v), x)
        F = ca.jacobian(self.m.measurement(x, v), v)
        c_offset_dynamics = self.m.ddyn(x, u, w, self.m.dt) - xp

        A_fun = ca.Function('A_fun', [x, u, w], [A])
        self.A_fun = ca.Function('A_fun_no_noise', [x, u], [A_fun(x, u, ca.DM.zeros(self.m.nw, 1))])
        B_fun = ca.Function('B_fun', [x, u, w], [B])
        self.B_fun = ca.Function('B_fun_no_noise', [x, u], [B_fun(x, u, ca.DM.zeros(self.m.nw, 1))])
        E_fun = ca.Function('E_fun', [x, u, w], [E])
        self.E_fun = ca.Function('E_fun_no_noise', [x, u], [E_fun(x, u, ca.DM.zeros(self.m.nw, 1))])
        C_fun = ca.Function('C_fun', [x, v], [C])
        self.C_fun = ca.Function('C_fun_no_noise', [x], [C_fun(x, ca.DM.zeros(self.m.nv, 1))])
        F_fun = ca.Function('F_fun', [x, v], [F])
        self.F_fun = ca.Function('F_fun_no_noise', [x], [F_fun(x, ca.DM.zeros(self.m.nv, 1))])

        c_offset_dynamics_fun = ca.Function('c_offset_dynamics_fun', [x, u, w, xp], [c_offset_dynamics])
        self.c_offset_dynamics_fun = ca.Function('c_offset_dynamics_fun_no_noise', [x, u, xp], 
                                                 [c_offset_dynamics_fun(x, u, ca.DM.zeros(self.m.nw, 1), xp)])

        # initialize the list of constraints
        self.current_iteration_scp['g_list'] = [self.m.g for _ in range(self.N)]
        self.current_iteration_scp['g_list'].append(self.m.gf)  # final state constraint

    def initialize_tube_cost_quadratization(self):
        nx = self.m.nx
        nu = self.m.nu
        nw = self.m.nw
        ny = self.m.ny
        nv = self.m.nv
        ni = self.m.ni
        ni_f = self.m.ni_f
        N = self.N

        # primal_x = ca.MX.sym('primal_x', nx, N + 1)  # nominal trajectory
        # primal_u = ca.MX.sym('primal_u', nu, N)  # nominal input trajectory
        primal_vec = ca.MX.sym('primal_vec', nx*(N+1) + nu*N + ni_f)
        Phi_xx = [[ca.MX.sym(f'Phi_xx_{k}_{j}', nx, nw) for j in range(N + 1)] for k in range(N+1)]
        Phi_ux = [[ca.MX.sym(f'Phi_ux_{k}_{j}', nu, nw) for j in range(N + 1)] for k in range(N+1)] # todo: only half of those are non-zeros
        Phi_xy = [[ca.MX.sym(f'Phi_xy_{k}_{j}', nx, nv) for j in range(N + 1)] for k in range(N+1)]
        Phi_uy = [[ca.MX.sym(f'Phi_uy_{k}_{j}', nu, nv) for j in range(N + 1)] for k in range(N+1)]

        # Create the function
        A_fun = self.A_fun
        B_fun = self.B_fun
        E_fun = self.E_fun
        F_fun = self.F_fun
        tube_cost_fun = self.fast_SLS_solver.initialize_tube_cost_fun(A_fun, B_fun, E_fun, F_fun)

        # Extract Phi_u blocks and convert to DM
        # M_vals = SLS.tensor3_list_to_vector(M)
        Phi_xx_vals = SLS.tensor3_list_to_vector(Phi_xx)
        Phi_ux_vals = SLS.tensor3_list_to_vector(Phi_ux)
        Phi_xy_vals = SLS.tensor3_list_to_vector(Phi_xy)
        Phi_uy_vals = SLS.tensor3_list_to_vector(Phi_uy)

        # cost = tube_cost_fun(primal_x, primal_u, *Phi_x_vals, *Phi_u_vals)
        cost = tube_cost_fun(primal_vec, *Phi_xx_vals, *Phi_ux_vals, *Phi_xy_vals, *Phi_uy_vals)
        # primal_vec = ca.vertcat(primal_x.reshape((-1, 1)), primal_u.reshape((-1, 1)))
        jac = ca.jacobian(cost, primal_vec).T
        # todo: use jtimes instead of jacobian, as it is more efficient
        
        H_reg = 0.01 * ca.DM.eye(primal_vec.shape[0])
        Hess, Hess_fun = ca.hessian(cost, primal_vec)
        H_tilde = Hess + H_reg
        # H_tilde = jac @ jac.T + H_reg # Newton-Gaussian approximation

        # Define input list
        inputs = [primal_vec] + [ca.vertcat(*[ca.reshape(m, -1, 1) for m in Phi_xx_vals])] + \
                                [ca.vertcat(*[ca.reshape(m, -1, 1) for m in Phi_ux_vals])] + \
                                [ca.vertcat(*[ca.reshape(m, -1, 1) for m in Phi_xy_vals])] + \
                                [ca.vertcat(*[ca.reshape(m, -1, 1) for m in Phi_uy_vals])]

        # Create CasADi function
        self.tube_cost_linear_fun = ca.Function(
            'tube_cost_linear_term',
            inputs,
            [jac]
        )
        self.tube_cost_quad_fun = ca.Function(
            "tube_cost_quad_term",
            inputs, 
            [H_tilde]
        )

    def update_traj_cost_wrt_tube(self):
        Phi_xx_current = self.current_iteration_scp['Phi_xx']
        Phi_ux_current = self.current_iteration_scp['Phi_ux']
        Phi_xy_current = self.current_iteration_scp['Phi_xy']
        Phi_uy_current = self.current_iteration_scp['Phi_uy']
        primal_x_current = self.current_iteration_scp['primal_x']
        primal_u_current = self.current_iteration_scp['primal_u']
        primal_vec_current = self.current_iteration_scp['primal_vec']
        
        # TODO: check which one is correct (note: ca.reshape is column-wise)
        # BTW: this may be a potential reason for adjoint correction result mismatch
        # Phi_xx_vec = SLS.tensor4_to_vector(Phi_xx_current)
        # Phi_ux_vec = SLS.tensor4_to_vector(Phi_ux_current)
        # Phi_xy_vec = SLS.tensor4_to_vector(Phi_xy_current)
        # Phi_uy_vec = SLS.tensor4_to_vector(Phi_uy_current)
        def tensor4_to_vector(tensor:np.ndarray):
            return tensor.transpose((0,1,3,2)).flatten()
        Phi_xx_vec = tensor4_to_vector(Phi_xx_current)
        Phi_ux_vec = tensor4_to_vector(Phi_ux_current)
        Phi_xy_vec = tensor4_to_vector(Phi_xy_current)
        Phi_uy_vec = tensor4_to_vector(Phi_uy_current)

        H0_jacobian = self.tube_cost_linear_fun(
            # primal_x_current,
            # primal_u_current,
            primal_vec_current,
            Phi_xx_vec,
            Phi_ux_vec,
            Phi_xy_vec, 
            Phi_uy_vec
        )
        self.fast_SLS_solver.add_linear_cost(H0_jacobian)

        H0_hess = self.tube_cost_quad_fun(
            # primal_x_current,
            # primal_u_current,
            primal_vec_current,
            Phi_xx_vec,
            Phi_ux_vec,
            Phi_xy_vec, 
            Phi_uy_vec
        )
        
        size = self.N * (self.m.nx + self.m.nu) + self.m.nx
        ni_f = self.m.ni_f
        if ni_f > 0:
            H_reg = ca.diagcat(ca.DM.eye(size), ca.DM.zeros(ni_f, ni_f))
        else:
            H_reg = ca.DM.eye(size)
        self.fast_SLS_solver.update_quadratic_cost(H0_hess + self.H_reg_coef * H_reg)

    def update_jacobian(self, obstacles=[], cur_iter=0):
        """
        This method updates the Jacobian matrices
        :return:
        """
        # update the Jacobian matrices
        nx = self.m.nx
        nu = self.m.nu
        nw = self.m.nw
        nv = self.m.nv
        ny = self.m.ny
        N = self.N

        primal_x = self.current_iteration_scp['primal_x']
        primal_u = self.current_iteration_scp['primal_u']
        # initialize A_list as a list of zeros nx x nx matrices
        A_list = [np.zeros((nx, nx)) for _ in range(N)]
        B_list = [np.zeros((nx, nu)) for _ in range(N)]
        C_list = [np.zeros((ny, nx)) for _ in range(N)]
        c_offset_list = [np.zeros((nx, 1)) for _ in range(N)]
        E_list = [np.zeros((nx, nw)) for _ in range(N + 1)]
        F_list = [np.zeros((nx, nv)) for _ in range(N + 1)]

        if self.linearization_error:
            raise NotImplementedError("Linearization error is not implemented yet")
        else:
            E_init = getattr(self.m, 'E_init', np.array(self.E_fun(primal_x[:,0], primal_u[:,0])))
            E_list[0] = E_init
            F_init = getattr(self.m, 'F_init', np.array(self.F_fun(primal_x[:,0])))
            F_list[0] = F_init
            for i in range(1, N):
                E_list[i] = np.array(self.E_fun(primal_x[:, i], primal_u[:, i]))
                F_list[i] = np.array(self.F_fun(primal_x[:, i-1]))
            E_list[N] = np.array(self.E_fun(primal_x[:, N], np.zeros((nu))))
            F_list[N] = np.array(self.F_fun(primal_x[:, N-1]))

        for i in range(N):
            A_list[i] = np.array(self.A_fun(primal_x[:, i], primal_u[:, i]))
            B_list[i] = np.array(self.B_fun(primal_x[:, i], primal_u[:, i]))
            C_list[i] = np.array(self.C_fun(primal_x[:, i]))
            c_offset_list[i] = self.c_offset_dynamics_fun(primal_x[:, i], primal_u[:, i], primal_x[:, i + 1])
        
        if cur_iter == 0:
            self.initialize_M(A_list, B_list, C_list, E_list, F_list)
        self.current_iteration_scp['A_list'] = A_list
        self.current_iteration_scp['B_list'] = B_list
        self.A_list = A_list
        self.B_list = B_list
        self.current_iteration_scp['c_offset_list'] = c_offset_list
        self.current_iteration_scp['E_list'] = E_list
        self.C_list = C_list
        self.F_list = F_list
        self.current_iteration_scp['C_list'] = C_list
        self.current_iteration_scp['F_list'] = F_list

        ni = self.m.ni
        g_list = [np.zeros(ni) for _ in range(N)]
        # update the dynamics list of constraints
        for i in range(N):
            z = ca.vertcat(primal_x[:, i], primal_u[:, i])
            g_list[i] = self.m.g - self.m.G @ z
        # the last element of g_list is the final state constraint
        z = ca.vertcat(primal_x[:, -1])
        g_list.append(self.m.gf - self.m.Gf @ z)

        self.current_iteration_scp['g_list'] = g_list
        self.current_iteration_scp['c_offset_list'] = c_offset_list
        self.fast_SLS_solver.update_dynamics_list(A_list, B_list, E_list, g_list, c_offset_list, 
                                                  obstacles, self.current_iteration_scp["primal_x"][:2,:], 
                                                  cur_iter < 5)
        self.fast_SLS_solver.update_observer_list(self.C_list, self.F_list)

        # update the linear part of the cost
        H_mat = self.H_mat
        q_lin_cost = 2 * H_mat @ (self.current_iteration_scp['primal_vec'] - self.goal_offset)
        q_lin_cost[-self.m.ni_f:] = ca.DM.ones(self.m.ni_f, 1)
        self.fast_SLS_solver.update_linear_cost(q_lin_cost)
        
        if self.state_dependent:
            self.update_traj_cost_wrt_tube()

    def initialize_fast_SLS_solver(self, obstacles):
        """
        This method initializes the fast-SLS solver
        :return:
        """
        init_LTV = LTV_OF(self.m, self.N)
        self.fast_SLS_solver = fast_OF_SLS_cvxpy(self.N, self.Q, self.R, init_LTV, self.Qf, 
                                               self.Q_reg, self.R_reg, self.Q_reg_f, 
                                               parallel=self.parallel, obstacles=obstacles)
        
        self.H_mat = self.fast_SLS_solver.solver_forward.H_mat
 
    def socp_step(self, obstacles=[]):
        """ This method performs one step of the SCP algorithm, which is a fast-SLS step.
        :return: True if the step was successful, False otherwise
        """

        delta_x0 = np.zeros(self.m.nx) # not state perturbation here
        solution = self.fast_SLS_solver.solve(delta_x0, obstacles, self.current_iteration_scp["primal_x"][:2,:])

        if solution['success']:
            # extract nominal trajectory
            delta_x = np.copy(solution['primal_x'])
            delta_u = np.copy(solution['primal_u'])

            delta_vec = np.concatenate([delta_x.flatten(), delta_u.flatten()])
            dual_vec = solution['dual_vec']
            primal_delta_vec = np.copy(solution['primal_vec'])

            self.current_iteration_scp['primal_x'] = self.current_iteration_scp['primal_x'] + delta_x
            self.current_iteration_scp['primal_u'] = self.current_iteration_scp['primal_u'] + delta_u
            self.current_iteration_scp['dual_vec'] = dual_vec
            self.current_iteration_scp['primal_vec'] += primal_delta_vec
            self.current_iteration_scp['dual_mu'] = solution['dual_mu']
            self.current_iteration_scp['dual_mu_f'] = solution['dual_mu_f']

            self.current_iteration_scp['dual_eta'] = solution['eta']
            self.current_iteration_scp['dual_eta_f'] = solution['eta_f']

            self.current_iteration_scp['delta_vec'] = delta_vec
            self.current_iteration_scp['backoff'] = solution['backoff']
            self.current_iteration_scp['backoff_x'] = solution['backoff_x']
            self.current_iteration_scp['backoff_u'] = solution['backoff_u']
            self.current_iteration_scp['backoff_f'] = solution['backoff_f']
            self.current_iteration_scp['Phi_xx_mat'] = solution.get('Phi_xx_mat')
            self.current_iteration_scp['Phi_ux_mat'] = solution.get('Phi_ux_mat')
            self.current_iteration_scp['Phi_xy_mat'] = solution.get('Phi_xy_mat')
            self.current_iteration_scp['Phi_uy_mat'] = solution.get('Phi_uy_mat')
            self.current_iteration_scp['Phi_xx'] = solution.get('Phi_xx')
            self.current_iteration_scp['Phi_xy'] = solution.get('Phi_xy')
            self.current_iteration_scp['Phi_ux'] = solution.get('Phi_ux')
            self.current_iteration_scp['Phi_uy'] = solution.get('Phi_uy')
            self.current_iteration_scp['M'] = 0.

            self.current_iteration_scp['backoff_process'] = solution.get('backoff_process')
            self.current_iteration_scp['backoff_process_f'] = solution.get('backoff_process_f')
            self.current_iteration_scp['backoff_measurement'] = solution.get('backoff_measurement')
            self.current_iteration_scp['backoff_measurement_f'] = solution.get('backoff_measurement_f')

            primal_x = self.current_iteration_scp['primal_x']
            primal_u = self.current_iteration_scp['primal_u']
            x = primal_x[:, :-1]
            u = primal_u
            cost_x = np.sum(x * (self.Q @ x))
            cost_u = np.sum(u * (self.R @ u))
            x_f = primal_x[:, -1]
            cost_f = x_f @ self.Qf @ x_f
            self.current_iteration_scp['cost_nominal'] = cost_x + cost_u + cost_f
            self.current_iteration_scp['cost'] = self.current_iteration_scp['cost_nominal'] + \
                                                solution['cost_tube']
            self.current_iteration_scp["cost_tube"] = solution["cost_tube"]

        # reset fast_SLS solver
        self.fast_SLS_solver.reset_solver_to_zeros()

        if self.verbose:
            if not solution['success']:
                print('SCP-SLS: Fast-SLS did not converge!')

        return solution['success']
        # todo implement the M update

    def post_processing_solution(self):
        """
        This method post-processes the solution
        :return:
        """
        solution = self.current_iteration_scp.copy()
        solution.update(self.convergence_data_scp)
        solution['it_data'] = self.it_data.copy()

        return solution

    def reset(self):
        """
        This method resets the SCP_SLS solver to its initial state.
        :return:
        """
        self.current_iteration_scp = {}
        self.convergence_data_scp = {}
        self.it_data = {}
        self.fast_SLS_solver.reset_solver_to_zeros()
        self.initialize_jacobian_Function()
        self.initialize_fast_SLS_solver()
        self.initialize_nominal_trajectory_solver()

    @staticmethod
    def printHeader():
        fixed_width = 10
        # Format headers to have fixed width and right alignment
        headers = ["it (SCP)", "primal", "dual", "cost", "traj_cost", "tube_cost"]
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
        table.align["traj_cost"] = "right"
        table.align["tube_cost"] = "right"

        # Set a fixed width for all columns
        fixed_width = 10
        # Set fixed widths for each column individually
        table.max_width["it"] = fixed_width
        table.max_width["primal"] = fixed_width
        table.max_width["dual"] = fixed_width
        table.max_width["cost"] = fixed_width
        table.max_width["traj_cost"] = fixed_width
        table.max_width["tube_cost"] = fixed_width

        print(table.get_string(end=0))
        table.hrules = NONE

        return table

    def printLine(self, i, table):

        fixed_width = 10
        primal = np.max(self.current_iteration_scp['primal_vec'])
        dual = np.max(np.nan_to_num(self.current_iteration_scp['dual_vec']))
        cost = self.current_iteration_scp['cost']
        tube_cost = self.current_iteration_scp['cost_tube']
        traj_cost = self.current_iteration_scp["cost_nominal"]
        iteration_str = f"{i:>{fixed_width}}"
        primal_val = f"{float(primal):>{fixed_width}.2e}"
        dual_val = f"{float(dual):>{fixed_width}.2e}"
        cost_val = f"{float(cost):>{fixed_width}.2e}" 
        traj_val = f"{float(traj_cost):>{fixed_width}.2e}"
        tube_val = f"{float(tube_cost):>{fixed_width}.2e}"
        # todo add absolute primal and dual values

        table.add_row([iteration_str, primal_val, dual_val, cost_val, traj_val, tube_val])
        print(table.get_string(start=len(table._rows) - 1, end=len(table._rows), header=False))

    def check_convergence_scp(self):
        """
        This method checks the convergence of the SCP algorithm
        :return: True if the SCP algorithm has converged
        """
        # check if the value delta_x is already assigned
        if 'delta_vec' in self.current_iteration_scp:
            delta_vec = self.current_iteration_scp['delta_vec']
            if np.linalg.norm(delta_vec) < self.epsilon_convergence:
                return True
        return False
