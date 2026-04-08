from dyn.model import Model
from dyn.G1.G1 import G1
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


class G1OF(G1):
    def __init__(self):
        G1.__init__(self)
        self.ny = 30
        self.nv = 30

        self.C = np.eye(self.nx)[:self.ny]
        # self.F = 1e-5 * np.eye(self.ny)
        self.F = 1e-3 * np.eye(self.ny) 

    def measurement(self, x, v):
        measurement = self.C @ x + self.F @ v

        return measurement
