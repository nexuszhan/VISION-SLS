import casadi as ca
import numpy as np

from solver.ocp import OCP
from dyn.LTV import LTV
from dyn.LTI import LTI
from scipy.linalg import block_diag


EPSILON = 1e-10  # small value to avoid numerical issues

class QP(OCP):
    def __init__(self, N, Q, R, m, Qf, obstacles=[]):
        super().__init__(N, Q, R, m, Qf)
        # self.qp_solver_name = 'qpoases'
        # self.nominal_solver_options = {
        #     'printLevel': 'none',
        #     'terminationTolerance': 1e-4,
        #     # 'constraintTolerance': 1e-4,
        #     'boundTolerance': 1e-6
        #     }  
        self.qp_solver_name = 'osqp' # OSQP is faster
        self.nominal_solver_options = {
            'osqp': {'verbose': False, 'polish': True, 'adaptive_rho': True, #'polish_refine_iter': 20
                    #  'scaled_termination': False, 'check_termination': 1
                    #  'max_iter': 50000, 'eps_abs': 1e-9, 'eps_rel': 1e-9
                     },
            'warm_start_primal': False 
            } 
        self.epsilon_backoff = 1e-10
        self.solution = {}
        self.solver_params = {}
        self.verbose = True

        # dynamics matrices
        self.initialize_list_dynamics()

        # solver matrices
        self.A_mat = None
        self.H_mat = None
        self.nominal_ubg = None  # nominal upper bound on the inequality constraints (without backoff)
        self.ubg = None
        # self.initialize_dynamics_and_cost_forward_solver()

        # solver objects
        self.solver_qp = None
        self.ubx_fun = None
        self.lbx_fun = None
        self.lbg = None
        self.q_cost_lin = None
        self.q_cost_quad = None
        self.init_guess = None
        self.initialize_solver(obstacles)

    def solve(self, x0, obstacles=[]):
        """
        This method solves the forward pass of the fast-SLS algorithm. In particular, it solves a linear trajectory
        optimization problem. For each iteration of the fast-SLS algorithm, it solves the problem with different
        backoff.
        :param x0: initial conditions
        :return:
        """
        nx = self.m.nx
        nu = self.m.nu
        ni = self.m.ni
        ni_f = self.m.ni_f
        N = self.N

        # solve the optimization problem
        # catch infeasible solution
        try:
            sol = self.solver_qp(a=self.A_mat, h= 2 * self.H_mat + self.q_cost_quad, lba=self.lbg, uba=self.ubg,
                                 g= self.q_cost_lin, lbx=self.lbx_fun(x0), ubx=self.ubx_fun(x0))
        except Exception as e:
            if self.verbose:
                print(e)
                print('QP: Infeasible forward solution. Try with another initial condition.')
            self.solution['success'] = False
            return self.solution

        self.solution['success'] = True
        # extract the solution
        # primal solution
        main_len = (nx + nu) * N + nx
        primal_vec = np.array(sol['x']).reshape(-1, 1)
        primal_y = np.reshape(np.concatenate([primal_vec[:main_len], np.zeros((nu, 1))]), (nx + nu, N + 1), order='F')
        primal_x = primal_y[:nx, :]
        primal_u = primal_y[nx:, :N]

        # Extract dual variables (Lagrange multipliers)
        dual_lam_a = sol['lam_a']

        if len(obstacles):
            dual_mu_terminal = np.array(dual_lam_a[-ni_f-N*len(obstacles):-N*len(obstacles)])
            dual_non_terminal = dual_lam_a[:-ni_f-N*len(obstacles)]
        else:
            # Separate terminal inequality constraint duals (last ni_f entries)
            dual_mu_terminal = np.array(dual_lam_a[-ni_f:])

            # Remaining duals include dynamics (nx) + stage inequality constraints (ni) for each step
            dual_non_terminal = dual_lam_a[:-ni_f]

        # Reshape to [N, nx + ni] to split dynamics vs. inequality constraints
        dual_non_terminal = np.reshape(dual_non_terminal, (N, nx + ni))

        # Extract only inequality constraint duals at each stage
        dual_mu_stage = dual_non_terminal[:, nx:]  # shape: [N, ni]


        # update the current iteration data
        self.solution['primal_vec'] = np.array(sol['x'])
        self.solution['primal_x'] = primal_x
        self.solution['primal_u'] = primal_u
        self.solution['slack_f'] = primal_vec[main_len:]

        self.solution['dual_vec'] = np.array(sol['lam_a'])
        self.solution['dual_mu'] = dual_mu_stage.T  # transpose to have the same index ordering as other time series
        self.solution['dual_mu_f'] = dual_mu_terminal.T  # transpose to have the same index ordering as other time series

        self.solution['cost'] = np.double(sol['cost'])
        
        self.init_guess = sol['x']

        return self.solution
    
    def init_A_mat(self, obstacles=[]):
        m = self.m

        nx = m.nx
        nu = m.nu
        ni = m.ni
        ni_f = m.ni_f 
        N = self.N

        A = np.ones((nx, nx))
        B = np.ones((nx, nu))
        G = np.ones((ni, nx + nu))
        Gf = np.ones((ni_f, nx))

        def x_off(t): return t*(nx+nu) 
        def u_off(t): return t*(nx+nu) + nx  

        base_rows = N*(nx + ni)
        total_rows = base_rows + ni_f + (len(obstacles) * N)
        total_cols = (N+1)*nx + N*nu + ni_f

        A_mat = np.zeros((total_rows, total_cols))
        I = np.eye(nx)
        Zero = np.zeros((ni, nx))

        for k in range(N):
            r0 = k*(nx + ni) 

            A_mat[r0:r0+nx, x_off(k):x_off(k)+nx] = A
            A_mat[r0:r0+nx, u_off(k):u_off(k)+nu] = B
            A_mat[r0:r0+nx, x_off(k+1):x_off(k+1)+nx] = -I
            
            A_mat[r0+nx:r0+nx+ni, x_off(k):x_off(k)+nx+nu] = G

        A_mat[base_rows:base_rows+ni_f, x_off(N):x_off(N)+nx] = Gf
        if ni_f:
            A_mat[base_rows:base_rows+ni_f, -ni_f:] = -np.eye(ni_f)

        r = base_rows + ni_f
        for _ in obstacles:
            for t in range(1, N+1): 
                A_mat[r, x_off(t) + 0] = 1.0
                A_mat[r, x_off(t) + 1] = 1.0
                r += 1
        
        A_mat = ca.DM(A_mat)
        return A_mat

    def initialize_solver(self, obstacles=[]):
        """
        This method initializes the solver for the forward pass of the fast-SLS algorithm.
        In particular, it creates a Casadi "conic" object that contains the structure of the optimization problem.
        Additionally, it initializes the lower and upper bounds on the inequality constraints.
        This function essentially defines the size of the problem and the structure of the optimization problem.
        """
        m = self.m

        nx = m.nx
        nu = m.nu
        ni = m.ni
        ni_f = m.ni_f  # number of terminal inequality constraints
        N = self.N

        A = np.ones((nx, nx))
        B = np.ones((nx, nu))
        G = np.ones((ni, nx + nu))
        Gf = np.ones((ni_f, nx))

        A_mat = np.zeros((0, nx))
        I = np.eye(nx)
        Zero = np.zeros((ni, nx))

        A_mat = self.init_A_mat(obstacles)
        A_mat = ca.sparsify(A_mat)

        Q = np.ones((nx, nx))
        R = np.ones((nu, nu))
        Qf = np.ones((nx, nx))
        # todo: don't define the cost here, as it should be defined outside the solver

        S_cost = block_diag(Q, R)
        H_mat = block_diag(ca.kron(ca.DM.eye(self.N), S_cost), Qf, np.zeros((ni_f, ni_f)))
        H_mat = ca.DM(H_mat)

        options = self.nominal_solver_options
        self.solver_qp = ca.conic('solver', self.qp_solver_name,
                                  {'a': A_mat.sparsity(), 'h': H_mat.sparsity()},
                                  options)

        self.lbg = ca.vertcat(ca.kron(ca.DM.ones(self.N, 1), ca.vertcat(ca.DM.zeros(m.nx, 1), -ca.DM.inf(ni,
                                                                                                         1))))
        self.lbg = ca.vertcat(self.lbg, -ca.DM.inf(ni_f, 1))
        self.lbg = ca.vertcat(self.lbg, -ca.DM.inf(self.N*len(obstacles), 1))

        x0_sym = ca.SX.sym('x0', nx)
        ubx = ca.vertcat(x0_sym + self.epsilon_backoff, ca.DM.inf(N * (nx + nu) + ni_f, 1))
        lbx = ca.vertcat(x0_sym - self.epsilon_backoff,
                         -ca.DM.inf(N * (nx + nu), 1),
                         ca.DM.zeros(ni_f, 1))

        self.ubx_fun = ca.Function('ubx_fun', [x0_sym], [ubx])
        self.lbx_fun = ca.Function('lbx_fun', [x0_sym], [lbx])

        # initialize the linear part of the cost to zeros
        self.reset_q_cost_lin()
        self.reset_q_cost_quad()

        # initialize the cost matrices
        Q = ca.DM(self.Q)
        R = ca.DM(self.R)
        Qf = ca.DM(self.Qf)

        S_cost = block_diag(Q, R)
        H_mat = block_diag(ca.kron(ca.DM.eye(self.N), S_cost), Qf, ca.DM.zeros(ni_f, ni_f))
        H_mat = ca.DM(H_mat)
        self.H_mat = H_mat
        
        self.init_guess = np.zeros(((N+1)*nx + N*nu + ni_f))

        return self.solver_qp
    
    def reconstruct_A_mat(self, A_list, B_list, G, Gf, obst_centers, obstacles, primal_pos, loose_terminal):
        m = self.m

        nx = m.nx
        nu = m.nu
        ni = m.ni
        ni_f = m.ni_f  
        N = self.N

        def x_off(t): return t*(nx+nu)
        def u_off(t): return t*(nx+nu) + nx 

        base_rows = N*(nx + ni)
        total_rows = base_rows + ni_f + (len(obstacles) * N)
        total_cols = (N+1)*nx + N*nu + ni_f

        A_mat = np.zeros((total_rows, total_cols))
        I = np.eye(nx)
        Zero = np.zeros((ni, nx))

        for k in range(N):
            A = A_list[k]
            B = B_list[k]

            r0 = k*(nx + ni)

            A_mat[r0:r0+nx, x_off(k):x_off(k)+nx] = A
            A_mat[r0:r0+nx, u_off(k):u_off(k)+nu] = B
            A_mat[r0:r0+nx, x_off(k+1):x_off(k+1)+nx] = -I
            
            A_mat[r0+nx:r0+nx+ni, x_off(k):x_off(k)+nx+nu] = G

        A_mat[base_rows:base_rows+ni_f, x_off(N):x_off(N)+nx] = Gf
        if ni_f and loose_terminal:
            A_mat[base_rows:base_rows+ni_f, -ni_f:] = -np.eye(ni_f)

        r = base_rows + ni_f
        for center in obst_centers:
            for t in range(1, N+1): 
                dist =  np.linalg.norm(primal_pos[:,t] - center) + 1e-5
                A_mat[r, x_off(t)] = -(primal_pos[0,t]-center[0]) / dist
                A_mat[r, x_off(t)+1] = -(primal_pos[1,t]-center[1]) / dist
                r += 1
        
        A_mat = ca.DM(A_mat)
        return A_mat

    def initialize_dynamics_and_cost_forward_solver(self, obstacles=[], primal_pos=[], loose_terminal=False):
        """
        Constructs a big matrix from a list of A and B matrices.
        :return: The constructed big matrix.
        """
        A_list = self.A_list
        B_list = self.B_list

        nx = self.m.nx
        nu = self.m.nu
        ni = self.m.ni
        ni_f = self.m.ni_f
        N = self.N
        m = self.m
        
        primal_pos_DM = primal_pos
        obst_centers = [o[0] for o in obstacles]
        obst_rads = [o[1] for o in obstacles]
        # initialization of the A matrix (dynamics and inequality constraints)
        A_mat = ca.DM.zeros((0, nx))
        I = ca.DM.eye(nx)
        Zero = ca.DM.zeros((ni, nx))

        G_Zero = ca.horzcat(self.m.G, Zero)

        A_mat = self.reconstruct_A_mat(A_list, B_list, m.G, m.Gf, obst_centers, obstacles, primal_pos, loose_terminal)
        
        self.A_mat = A_mat
        
        if isinstance(self.m, LTI): # assume time invariant constraints
            nominal_ubg = (
                ca.kron(ca.DM.ones(self.N, 1), ca.vertcat(ca.DM.zeros(m.nx, 1), m.g - self.epsilon_backoff)))
            nominal_ubg = ca.vertcat(nominal_ubg, m.gf - self.epsilon_backoff)
        elif isinstance(self.m, LTV): # assume time-varying constraints
            nominal_ubg = ca.vertcat(*[ca.vertcat(ca.DM.zeros(m.nx, 1), g - self.epsilon_backoff) for g in m.g_list[:-1]])
            nominal_ubg = ca.vertcat(nominal_ubg, m.g_list[-1] - self.epsilon_backoff)
            # todo: is this correct? Is m.g_list updated in the LTV model?
            collision_avoidance_ubg = []
            for center, rad in zip(obst_centers, obst_rads):
                for t in range(1, self.N+1):                                  
                    # collision_avoidance_ubg.append( ca.sqrt(ca.sumsqr(primal_pos_DM[:,t] - center + 1e-8)) - rad )
                    collision_avoidance_ubg.append( np.linalg.norm(primal_pos[:,t] - center) - rad )
            nominal_ubg = ca.vertcat(nominal_ubg, *collision_avoidance_ubg)
        else:
            # not the correct instance
            raise ValueError('The model should be either LTI or LTV')

        self.nominal_ubg = nominal_ubg
        self.ubg = nominal_ubg

        self.init_guess = np.zeros(((N+1)*nx + N*nu + ni_f))

    def update_dynamics(self, new_A_list, new_B_list, new_E_list=None, new_g_list=None, 
                        obstacles=[], primal_pos=[], loose_terminal=False):
        """
        Update the dynamics matrices A and B.
        :param new_A_list: new list of A matrices
        :param new_B_list: new list of B matrices
        :return:
        """
        assert len(new_A_list) == self.N, f"new_A_list should have length {self.N}, but got {len(new_A_list)}"
        assert len(new_B_list) == self.N, f"new_B_list should have length {self.N}, but got {len(new_B_list)}"


        self.A_list = new_A_list
        self.B_list = new_B_list
        self.m.A_list = new_A_list
        self.m.B_list = new_B_list

        if new_E_list is not None:
            assert len(new_E_list) == self.N +1, f"new_E_list should have length {self.N}, but got {len(new_E_list)}"
            self.E_list = new_E_list
            self.m.E_list = new_E_list

        if new_g_list is not None:
            assert len(new_g_list) == self.N + 1, f"new_g_list should have length {self.N + 1}, but got {len(new_g_list)}"
            self.g_list = new_g_list
            self.m.g_list = new_g_list

        # reinitialize the dynamics and cost forward solver
        self.initialize_dynamics_and_cost_forward_solver(obstacles, primal_pos, loose_terminal)

    def update_ubg(self, new_ubg):
        """
        Update the upper bound on the inequality constraints.
        :param new_ubg: new upper bound on the inequality constraints
        :return:
        """
        self.ubg = new_ubg

    def reset_ubg(self):
        """
        Reset the upper bound on the inequality constraints to the nominal value.
        :return:
        """
        self.ubg = self.nominal_ubg

    def offset_constraints(self, offset_equality_constraints, obstacles):
        """
        Offset the upper bound on the equality constraints.
        :param offset_equality_constraints: value to offset the upper bound on the equality constraints
        :return:
        """
        # offset is a matrix of shape (nx, N)
        assert offset_equality_constraints.shape == (self.m.nx, self.N), \
            f"offset should have shape {(self.m.nx, self.N)}, but got {offset_equality_constraints.shape}"

        # construct a vector of the same shape as self.ubg
        # initialize offset_vector
        offset_vector = ca.DM.zeros((0, 1))
        for i in range(self.N):
            offset_i = ca.DM(offset_equality_constraints[:, i])
            zeros_offset_inequalities = ca.DM.zeros((self.m.ni, 1)) # assumes constant number of inequality constraints
            offset_vector = ca.vertcat(offset_vector, offset_i, zeros_offset_inequalities)

        # add the zeros terminal offset
        offset_vector = ca.vertcat(offset_vector, ca.DM.zeros((self.m.ni_f, 1)))
        offset_vector = ca.vertcat(offset_vector, ca.DM.zeros((self.N*len(obstacles), 1)))

        self.ubg = self.ubg - offset_vector + EPSILON  # add a small slack to avoid infeasibility
        self.lbg = self.lbg - offset_vector  # add a small slack to avoid infeasibility

        # convert ubg and lbg to Casadi DM vectors if not already
        if not isinstance(self.ubg, ca.DM):
            self.ubg = ca.DM(self.ubg)
        if not isinstance(self.lbg, ca.DM):
            self.lbg = ca.DM(self.lbg)

    def update_q_cost_lin(self, q_cost_lin):
        """
        Update the linear part of the cost.
        :param q_cost_lin: new linear part of the cost
        :return:
        """
        assert q_cost_lin.shape[0] == (self.m.nx + self.m.nu) * self.N + self.m.nx + self.m.ni_f, \
            f"q_cost_lin should have shape {((self.m.nx + self.m.nu) * self.N + self.m.nx + self.m.ni_f, 1)}, but got {q_cost_lin.shape}"
        # convert to Casadi DM if not already
        if not isinstance(q_cost_lin, ca.DM):
            q_cost_lin = ca.DM(q_cost_lin)

        self.q_cost_lin = q_cost_lin

    def add_q_cost_lin(self, q_cost_lin):
        """
        Add to the linear part of the cost.
        :param q_cost_lin: new linear part of the cost
        :return:
        """
        assert q_cost_lin.shape[0] == (self.m.nx + self.m.nu) * self.N + self.m.nx + self.m.ni_f, \
            f"q_cost_lin should have shape {((self.m.nx + self.m.nu) * self.N + self.m.nx + self.m.ni_f, 1)}, but got {q_cost_lin.shape}"
        # convert to Casadi DM if not already
        if not isinstance(q_cost_lin, ca.DM):
            q_cost_lin = ca.DM(q_cost_lin)

        self.q_cost_lin += q_cost_lin


    def reset_q_cost_lin(self):
        """
        Reset the linear part of the cost to zero.
        :return:
        """
        nx = self.m.nx
        nu = self.m.nu
        N = self.N
        self.q_cost_lin = ca.DM.zeros((nx + nu) * N + nx + self.m.ni_f)

    def update_q_cost_quad(self, q_cost_quad):
        self.q_cost_quad = ca.DM(q_cost_quad)
    
    def reset_q_cost_quad(self):
        nx = self.m.nx
        nu = self.m.nu
        N = self.N
        self.q_cost_quad = ca.DM.zeros((nx + nu) * N + nx + self.m.ni_f,
                                       (nx + nu) * N + nx + self.m.ni_f)
