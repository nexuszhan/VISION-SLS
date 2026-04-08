from dyn.model import Model
from dyn.light_dark import LightDark
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from util.minkowski_sum_polygons import minkowski_sum_polygons

class LightDark_OF(LightDark):
    def __init__(self):
        LightDark.__init__(self)
        self.ny = 2
        self.nv = 2

        self.C = np.eye(self.ny)
        self.F_init = 0.1 * np.eye(self.ny) # uncertainty on initial condition
    
    def measurement(self, x, v):
        return ca.vertcat(x[0]+0.02*(2-x[0])**2*v[0], x[1]+0.02*(2-x[0])**2*v[1])

    def plot_disturbance_map(self, ax):
        a = 0.02
        x_min, x_max = -2.0, 4.0
        y_min, y_max = -2.0, 4.0
        # Dense grid in x; replicate along y
        nx, ny = 300, 300
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y)
        F = a * (X - 2.0)**2
        cf = ax.imshow(F, origin="lower", extent=[x_min, x_max, y_min, y_max],
                aspect="auto", cmap="Greys")  
        return cf
    
    @staticmethod
    def combine_tubes(tubes_x, tubes_y):
        length = len(tubes_x)
        tubes = []
        for i in range(length):
            polygon_x, points_x = tubes_x[i]
            polygon_y, points_y = tubes_y[i]
            tube = minkowski_sum_polygons(points_x, points_y)
            points = np.array(tube.exterior.coords.xy).T
            # create a tuple of polygon and points
            combined_tube = (tube, points)
            tubes.append(combined_tube)

        return tubes