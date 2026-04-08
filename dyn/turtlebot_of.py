from dyn.model import Model
from dyn.turtlebot import Turtlebot
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import Variables, MathematicalProgram
import pydrake.symbolic as sym
from util.symbolic_utils import sym_to_pytorch
import os
import torch


class Turtlebot_OF(Turtlebot):
    def __init__(self):
        Turtlebot.__init__(self)
        self.ny = 3
        self.nv = 3

        self.C = np.array([[-0.1844,  0.0084, -0.9828, 0., 0.],
                           [-0.4349,  0.4429,  0.7840, 0., 0.],
                           [ 0.6103, -0.1929, -0.7683, 0., 0.]])
        # self.C = np.eye(self.nx)[:self.ny, :]
        self.F = 0.05 * np.eye(self.ny)

        base_path = os.path.dirname(__file__)
        loaded = np.load(os.path.join(base_path, "turtlebot_error_bound3_0.95_0.06.npz"),
                        allow_pickle=True)
        self.theta_poly, self.theta_bias, degree = loaded['theta_poly'][:,None], loaded['theta_bias'], loaded['degree']

        prog = MathematicalProgram()
        x_red = sym.MakeVectorContinuousVariable(loaded['nx'], "x_red")
        V = prog.NewFreePolynomial(Variables(x_red), degree)
        mons = list(V.monomial_to_coefficient_map().keys())
        mons_expr = np.array([mons[i].ToExpression() for i in range(len(mons))])
        self.mons_fn, _ = sym_to_pytorch(mons_expr, x_red)

        jac_mons_expr = np.array([mons_expr[i].Jacobian(x_red) for i in range(len(mons_expr))])
        self.jac_mons_fn, _ = sym_to_pytorch(jac_mons_expr, x_red)

    def measurement(self, x, v):
        # e = self.get_disturbance(x, 1.) * np.eye(self.nv)
        # measurement = self.C @ x + e @ v
        measurement = self.C @ x + self.F @ v

        return measurement
    
    def get_disturbance(self, x, scale=1.0):
        x0 = x[0]
        x1 = x[1]
        x2 = x[2]

        e = 1*self.theta_poly[0,0] + x2*self.theta_poly[1,0] + x2**2*self.theta_poly[2,0] + x2**3*self.theta_poly[3,0] + \
            x1*self.theta_poly[4,0] + x1*x2*self.theta_poly[5,0] + x1*x2**2*self.theta_poly[6,0] + x1**2*self.theta_poly[7,0] + \
            x1**2*x2*self.theta_poly[8,0] + x1**3*self.theta_poly[9,0] + x0*self.theta_poly[10,0] + x0*x2*self.theta_poly[11,0] + \
            x0*x2**2*self.theta_poly[12,0] + x0*x1*self.theta_poly[13,0] + x0*x1*x2*self.theta_poly[14,0] + x0*x1**2*self.theta_poly[15,0] + \
            x0**2*self.theta_poly[16,0] + x0**2*x2*self.theta_poly[17,0] + x0**2*x1*self.theta_poly[18,0] + x0**3*self.theta_poly[19,0]
        if isinstance(x, ca.MX):
            return ca.fmax(e, self.theta_bias) * scale
        else:
            return max(e, self.theta_bias) * scale
    
    def dcn(self, x):
        return x.detach().cpu().numpy()

    def cov_fn(self, x, scale=1.):
        if torch.is_tensor(x):
            mons_val = self.mons_fn(x).T
            return scale*(torch.tensor(self.theta_poly.T).float() @ mons_val).clip(min=torch.tensor(self.theta_bias).float()).T
        else:
            mons_val = self.dcn(self.mons_fn(torch.tensor(x).float()).T)
            return scale*(self.theta_poly.T @ mons_val).clip(min=self.theta_bias).T
    
    def plot_disturbance_map(self, ax):
        x_min, x_max = -0.5-3, 3.5-3
        y_min, y_max = -2.0, 2.0
        # Dense grid in x; replicate along y
        nx, ny = 300, 300
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y)
        grid = np.stack([X+3, Y], axis=-1)
        # grid_torch = torch.from_numpy(grid).float().view(-1, 2) 
        grid_torch = grid.astype(float).reshape(-1, 2)

        cf = ax.contourf(X, Y, self.cov_fn(grid_torch, 1.).reshape((nx,ny)), levels=10, cmap="Greys")
        
        return cf
        