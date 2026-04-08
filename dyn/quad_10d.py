import numpy as np
from dyn.model import Model
import casadi as ca
import matplotlib.pyplot as plt

class Quad10d(Model):
    def __init__(self):
        self.nx = 10
        self.nu = 3
        self.dt = 0.1

        self.G = ca.vertcat(np.eye(self.nx+self.nu), -np.eye(self.nx+self.nu))
        x_max = np.array([10., 10., 10., 5., 5., 5., 2., 2., 2., 2.]) # [x, y, z, vx, vy, vz, f, phi, theta, psi]
        x_min = np.array([-10., -10., 0., -5., -5., -5., -2., -2., -2., -2.])
        u_max = np.array([5., 5., 40.]) 
        u_min = np.array([-5., -5., 0.])
        x_max_f = x_max.copy() 
        x_min_f = x_min.copy()

        self.g = np.concatenate((x_max, u_max, -x_min, -u_min))
        self.ni = 2*(self.nx+self.nu)
        self.Gf = ca.vertcat(np.eye(self.nx), -np.eye(self.nx))
        self.gf = np.concatenate((x_max_f, -x_min_f))
        self.ni_f = 2*self.nx
        
        self.E = 0.01 * np.eye(self.nx)
        
        self.nw = self.nx

    def ode(self, X, u, w):
        dx, dy, dz = 0, 0, 0
        dvx, dvy, dvz = 3, 3, 1
        d1, d2, d3, d4 = 10, 10, 10, 10
        grav = 9.81
        kT = 1.
        n0 = 50.
        x_dot = X[3] - dx * X[0]
        y_dot = X[4] - dx * X[1]
        z_dot = X[5] - dx * X[2]
        vx_dot = grav * ca.tan(X[6]) - dvx * X[3]
        vy_dot = grav * ca.tan(X[7]) - dvy * X[4]
        vz_dot = kT * u[2] - grav - dvz * X[5]
        f_dot = X[8] - d1 * X[6]
        phi_dot = X[9] - d2 * X[7]
        theta_dot = n0 * u[0] - d3 * X[8]
        psi_dot = n0 * u[1] - d4 * X[9]
        Xd = ca.vertcat(x_dot, y_dot, z_dot, vx_dot, vy_dot, vz_dot, f_dot, phi_dot, theta_dot, psi_dot) + self.E @ w
        return Xd
    
    def replace_constraints(self, x_max, x_min, u_max, u_min, x_max_f, x_min_f):
        self.g = np.concatenate((x_max, u_max, -x_min, -u_min))
        self.gf = np.concatenate((x_max_f, -x_min_f))

    def plot_nominal_trajectory(self, X, ax=None):
        """
        Plot the nominal trajectory in the state space.
        
        Args:
            X: nominal trajectory (2 x N+1 array)
            ax: matplotlib axis (optional)
        
        Returns:
            matplotlib axis with the trajectory plot
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Plot the nominal trajectory
        ax.plot(X[0, :], X[1, :], 'r-', linewidth=1, label='Nominal trajectory')
        
        # Plot the initial condition
        ax.plot(X[0, 0], X[1, 0], 'bo', markersize=4, label='Initial state')
        
        # Plot the final condition
        ax.plot(X[0, -1], X[1, -1], 'go', markersize=4, label='Final state')

        plt.legend()
        
        return ax
    
    @staticmethod
    def plot_tube_as_rectangle(backoff, center, ax=None):
        """
        Plot bounding box as a rectangle.
        
        Args:
            backoff: backoff values (2D array)
            center: center point (2D array)
            ax: matplotlib axis (optional)
        
        Returns:
            matplotlib axis with the rectangle plot
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle as MatplotlibRectangle
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Unpack backoff and center
        width, height = 2*backoff[0], 2*backoff[1]
        x_center, y_center = center[0], center[1]

        # Calculate bottom-left corner based on center
        bottom_left_x = x_center - width / 2
        bottom_left_y = y_center - height / 2

        # Create a rectangle with the specified attributes
        rectangle = MatplotlibRectangle(
            (bottom_left_x, bottom_left_y),  # Bottom-left corner
            width,  # Width
            height,  # Height
            facecolor='lightgreen',  # Light green background
            edgecolor='green',  # Green edges
            linewidth=2,  # Edge line width
            alpha=0.3  # Transparency
        )

        ax.add_patch(rectangle)

        return ax