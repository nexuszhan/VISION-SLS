import casadi as ca
import numpy as np

from solver.ocp import OCP
from dyn.LTV import LTV


class NLP(OCP):
    def __init__(self, N, Q, R, m, Qf, obstacles=[], init_guess=None):
        super().__init__(N, Q, R, m, Qf)
        self.nlp_solver_name = 'ipopt'
        self.solution = {}
        self.solver_params = {}
        self.solver_nominal = {}
        # Solver options
        self.opts_ipopt = {
            'ipopt.print_level': 0,  # Minimal output
            'print_time': False,  # Disable timing info
            'ipopt.sb': 'yes'  # Silent barrier
        }
        self.verbose = False
        self.obstacles = obstacles
        self.init_guess = init_guess
        # self.initialize_solver(goals, obstacles)

    def solve(self, x0, goals):
        self.initialize_solver(goals, self.obstacles)

        # y0 = self.solver_params['y0']
        if self.init_guess is not None:
            y0 = self.init_guess
        else:
            if len(goals.shape) == 1:
                y0 = np.concatenate([np.concatenate([x0, *goals]), np.zeros(self.N*self.m.nu)]) 
            elif len(goals.shape) == 2:
                y0 = np.concatenate([goals.flatten(), np.zeros(self.N*self.m.nu)]) 
            else:
                raise NotImplementedError
        lbg = self.solver_params['lbg']
        ubg = self.solver_params['ubg']

        try:
            sol = self.solver_nominal(x0=y0, p=x0, lbg=lbg, ubg=ubg)
            
            # Check solver status
            solver_success = self._check_solver_status(sol)
            
            if solver_success:
                self.solution['success'] = True
                self._post_process_solution(sol)
            else:
                self.solution['success'] = False

        except Exception as e:
            self.solution['success'] = False
            print(e)
            return None

        return self.solution

    def _check_solver_status(self, sol):
        """
        Check the solver status and return True if successful, False otherwise.
        """
        try:
            stats = self.solver_nominal.stats()
            
            # Check if solution exists and solver converged
            if 'success' in stats:
                return bool(stats['success'])
            elif 'return_status' in stats:
                status = stats['return_status']
                # IPOPT successful statuses
                return status in ['Solve_Succeeded', 'Solved_To_Acceptable_Level']
            
            # Fallback: check if we have a valid solution
            return sol['x'] is not None and not ca.isnan(sol['f'])
        except Exception:
            # If we can't get stats, assume failure
            return False

    def _post_process_solution(self,sol):
        sol_vec = np.array(sol['x'])
        primal_x, primal_u, primal_vec = self._unpack_y(sol_vec)

        self.solution['primal_vec'] = primal_vec
        self.solution['primal_x'] = primal_x
        self.solution['primal_u'] = primal_u
        self.solution['dual_vec'] = np.array(sol['lam_g'])
        self.solution['cost'] = sol['f']

    def _unpack_y(self, y):
        """
        Given the stacked vector y = [ Z[:,0]; V[:,0];  … ; Z[:,N-1]; V[:,N-1]; Z[:,N] ],
        recover Z of shape (nx, N+1) and V of shape (nu, N).

        y may be a CasADi DM or a 1‑D numpy array/flat list.
        """
        # make sure it’s a flat numpy array
        nx = self.m.nx
        nu = self.m.nu
        N = self.N

        y_arr = np.array(y).flatten()

        sol_vec_x = y_arr[:nx * (N + 1)]
        sol_vec_u = y_arr[nx * (N + 1):]

        primal_x = sol_vec_x.reshape(nx, -1, order='F')
        primal_u = sol_vec_u.reshape(nu, -1, order='F')

        elems = []
        for i in range(N):
            elems.append(primal_x[:, i])
            elems.append(primal_u[:, i])
        elems.append(primal_x[:, N])  # final state

        primal_vec =  ca.vertcat(*elems)

        return primal_x, primal_u, primal_vec

    def initialize_solver(self, goals, obstacles):
        """
        This method initializes the nlp solver for nominal trajectory optimization
        :return:
        """
        N = self.N
        m = self.m

        G = self.m.G
        Gf = self.m.Gf
        g = self.m.g
        gf = self.m.gf

        Z = ca.MX.sym('state', self.m.nx, self.N + 1)
        V = ca.MX.sym('input', self.m.nu, self.N)
        p = ca.MX.sym('p', self.m.nx)

        g_eq = ca.vertcat()
        g_ineq = ca.vertcat()

        for i in range(N):  # Python uses 0-based indexing
            g_eq = ca.vertcat(g_eq, Z[:, i + 1] - m.ddyn(Z[:, i], V[:, i], h=self.m.dt))

        g_eq = ca.vertcat(g_eq, Z[:, 0] - p)

        for i in range(N):
            ineq = G @ ca.vertcat(Z[:, i], V[:, i]) - g
            g_ineq = ca.vertcat(g_ineq, ineq)

        ineq = Gf @ Z[:, -1] - gf
        g_ineq = ca.vertcat(g_ineq, ineq)

        # collision avoidance
        for obstacle in obstacles:
            center = ca.DM(obstacle[0])
            rad = obstacle[1]
            for t in range(1, N+1): # assume initial condition should not intersect
                ineq = -ca.sqrt(ca.sumsqr(Z[:2,t] - center) + 1e-8) + rad
                g_ineq = ca.vertcat(g_ineq, ineq)

        f = (Z[:, -1] - goals[-1]).T @ self.Qf @ (Z[:, -1] - goals[-1])
        for i in range(1, N):
            f += (Z[:, i] - goals[i]).T @ self.Q @ (Z[:, i] - goals[i]) + V[:, i-1].T @ self.R @ V[:, i-1]

        n_ineq = g_ineq.shape[0]
        lbg_ineq = [-ca.inf] * n_ineq
        ubg_ineq = np.zeros(n_ineq)

        n_eq = g_eq.shape[0]
        lbg_eq = np.zeros(n_eq)
        ubg_eq = np.zeros(n_eq)

        g = ca.vertcat(g_eq, g_ineq)
        y = ca.vertcat(ca.reshape(Z, (N + 1) * self.m.nx, 1), ca.reshape(V, N * self.m.nu, 1))

        # Define the NLP problem
        nlp = {
            'x': y,  # Decision variables
            'f': f,  # Objective function
            'p': p,
            'g': g
        }

        self.solver_params['lbg'] = ca.vertcat(lbg_eq, lbg_ineq)
        self.solver_params['ubg'] = ca.vertcat(ubg_eq, ubg_ineq)
        self.solver_params['y0'] = np.zeros((self.m.nx * (N + 1) + self.m.nu * N, 1))

        self.solver_nominal = ca.nlpsol('solver', self.nlp_solver_name, nlp, self.opts_ipopt)


