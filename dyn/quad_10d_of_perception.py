from dyn.model import Model
from dyn.quad_10d import Quad10d
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import Variables, MathematicalProgram
import pydrake.symbolic as sym
from util.symbolic_utils import sym_to_pytorch
import os
import torch

class Quad10d_OF(Quad10d):
    def __init__(self):
        Quad10d.__init__(self)
        self.ny = 5
        self.nv = 5

        self.C = np.array([[0.33892232, -0.78248858, 0.15168084, 0., 0., 0., -0.27302152, -0.41868305, 0., 0.],
                            [-0.47416508, -0.61890107, -0.22137581, 0., 0., 0., 0.23519689, 0.53647381, 0., 0.],
                            [0.78267038, -0.04408706, -0.32603681, 0., 0., 0., 0.42114031, 0.31909937, 0., 0.],
                            [0.06873424, -0.04491888, 0.89008391, 0., 0., 0., 0.4032591, 0.19593541, 0., 0.],
                            [0.21079303, 0.04344413, 0.17757122, 0., 0., 0., -0.72400945, 0.63083881, 0., 0.]])
        self.F = 0.05 * np.eye(self.ny)
        self.use_perception_model = True

        base_path = os.path.dirname(__file__)
        loaded = torch.load(os.path.join(base_path, "theta_lp_large_new2_val.pt"), map_location=torch.device('cpu'))
        self.theta_poly = self.dcn(loaded)
        self.theta_bias = 0.
        degree = 4

        prog = MathematicalProgram()
        x_red = sym.MakeVectorContinuousVariable(5, "x_red")
        V = prog.NewFreePolynomial(Variables(x_red), degree)
        mons = list(V.monomial_to_coefficient_map().keys())
        mons_expr = np.array([mons[i].ToExpression() for i in range(len(mons))])
        self.mons_fn, _ = sym_to_pytorch(mons_expr, x_red)

        self.poly_func = None
        self.initialize_poly_func()

    def measurement(self, x, v):
        e = self.get_disturbance(x, 1.)
        if isinstance(x, ca.MX):
            measurement = self.C @ x + ca.diag(e) @ v
        else:
            measurement = self.C @ x + np.diag(e) @ v

        return measurement
    
    def get_disturbance(self, x, scale=1.0):
        # mons_val = self.construct_poly(x)
        mons_val = self.poly_func(x)

        if isinstance(x, ca.MX):
            e = self.theta_poly.T @ mons_val
            return ca.fmax(ca.vertcat(*e), self.theta_bias) * scale
        else:
            e = self.theta_poly.T @ np.array(mons_val).squeeze()
            return np.maximum(e, self.theta_bias) * scale
    
    def dcn(self, x):
        return x.detach().cpu().numpy()

    def cov_fn(self, x, scale=1.):
        if torch.is_tensor(x):
            mons_val = self.mons_fn(x).T
            return scale*(torch.tensor(self.theta_poly.T).float() @ mons_val).clip(min=torch.tensor(self.theta_bias).float()).T
        else:
            mons_val = self.dcn(self.mons_fn(torch.tensor(x).float()).T)
            return scale*(self.theta_poly.T @ mons_val).clip(min=self.theta_bias).T
    
    def initialize_poly_func(self):
        x = ca.MX.sym("state", self.nx)
        # x0, x1, x2, x3, x4 = x[0] + 1.1, x[1] + 1.5, x[2] + 1.8, x[6], x[7]
        x0, x1, x2, x3, x4 = x[0] , x[1] , x[2] , x[6], x[7]
        self.poly_func = ca.Function("poly", [x], [1, x4, x4**2, x4**3, x4**4,
            x3, x3 * x4, x3 * x4**2, x3 * x4**3,
            x3**2, x3**2 * x4, x3**2 * x4**2,
            x3**3, x3**3 * x4, x3**4,
            x2, x2 * x4, x2 * x4**2, x2 * x4**3,
            x2 * x3, x2 * x3 * x4, x2 * x3 * x4**2,
            x2 * x3**2, x2 * x3**2 * x4, x2 * x3**3,
            x2**2, x2**2 * x4, x2**2 * x4**2,
            x2**2 * x3, x2**2 * x3 * x4, x2**2 * x3**2,
            x2**3, x2**3 * x4, x2**3 * x3, x2**4,
            x1, x1 * x4, x1 * x4**2, x1 * x4**3,
            x1 * x3,
            x1 * x3 * x4,
            x1 * x3 * x4**2,
            x1 * x3**2,
            x1 * x3**2 * x4,
            x1 * x3**3,
            x1 * x2,
            x1 * x2 * x4,
            x1 * x2 * x4**2,
            x1 * x2 * x3,
            x1 * x2 * x3 * x4,
            x1 * x2 * x3**2,
            x1 * x2**2,
            x1 * x2**2 * x4,
            x1 * x2**2 * x3,
            x1 * x2**3,
            x1**2,
            x1**2 * x4,
            x1**2 * x4**2,
            x1**2 * x3,
            x1**2 * x3 * x4,
            x1**2 * x3**2,
            x1**2 * x2,
            x1**2 * x2 * x4,
            x1**2 * x2 * x3,
            x1**2 * x2**2,
            x1**3,
            x1**3 * x4,
            x1**3 * x3,
            x1**3 * x2,
            x1**4,
            x0,
            x0 * x4,
            x0 * x4**2,
            x0 * x4**3,
            x0 * x3,
            x0 * x3 * x4,
            x0 * x3 * x4**2,
            x0 * x3**2,
            x0 * x3**2 * x4,
            x0 * x3**3,
            x0 * x2,
            x0 * x2 * x4,
            x0 * x2 * x4**2,
            x0 * x2 * x3,
            x0 * x2 * x3 * x4,
            x0 * x2 * x3**2,
            x0 * x2**2,
            x0 * x2**2 * x4,
            x0 * x2**2 * x3,
            x0 * x2**3,
            x0 * x1,
            x0 * x1 * x4,
            x0 * x1 * x4**2,
            x0 * x1 * x3,
            x0 * x1 * x3 * x4,
            x0 * x1 * x3**2,
            x0 * x1 * x2,
            x0 * x1 * x2 * x4,
            x0 * x1 * x2 * x3,
            x0 * x1 * x2**2,
            x0 * x1**2,
            x0 * x1**2 * x4,
            x0 * x1**2 * x3,
            x0 * x1**2 * x2,
            x0 * x1**3,
            x0**2,
            x0**2 * x4,
            x0**2 * x4**2,
            x0**2 * x3,
            x0**2 * x3 * x4,
            x0**2 * x3**2,
            x0**2 * x2,
            x0**2 * x2 * x4,
            x0**2 * x2 * x3,
            x0**2 * x2**2,
            x0**2 * x1,
            x0**2 * x1 * x4,
            x0**2 * x1 * x3,
            x0**2 * x1 * x2,
            x0**2 * x1**2,
            x0**3,
            x0**3 * x4,
            x0**3 * x3,
            x0**3 * x2,
            x0**3 * x1,
            x0**4
        ])
