import numpy as np
from dyn.LTI import LTI
from dyn.LTV import LTV
from dyn.LTI_OF import LTI_OF
from dyn.LTV_OF import LTV_OF

from solver.ocp import OCP


class OCP_OF(OCP):
    def __init__(self, N, Q, R, m, Qf, Q_reg=None, R_reg=None, Q_reg_f=None):
        super().__init__(N, Q, R, m, Qf, Q_reg, R_reg, Q_reg_f)
        self.C_list = None
        self.F_list = None

    def initialize_list_dynamics(self):
        """
        Initialize the linear dynamics.
        :return:
        """
        # initialize A, B, E
        super().initialize_list_dynamics()
        m = self.m

        if isinstance(m, LTI_OF):
            self.C_list = [m.C for _ in range(self.N)]
            self.F_list = [m.F for _ in range(self.N + 1)]

        elif isinstance(m, LTV_OF):
            self.C_list = m.C_list
            self.F_list = m.F_list
        else:
            raise ValueError('Model type not supported')

    # @staticmethod
    # def kalman_step(A, C, Q, R, S):
    #     # Predict the covariance
    #     S_pred = A @ S @ A.T + Q
    #     # Kalman gain
    #     L = np.linalg.solve(C @ S_pred @ C.T + R, S_pred @ C.T).T  # Equivalent to: (S_pred @ C.T @ np.linalg.inv(...))
    #     S_new = S_pred - L @ C @ S_pred
    #     return L, S_new


