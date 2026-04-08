import numpy as np
from dyn.model import Model
import casadi as ca
import matplotlib.pyplot as plt

class Turtlebot(Model):
    def __init__(self):
        self.nx = 5
        self.nu = 2
        self.dt = 0.1
        self.discretization_method = "euler"

        self.G = ca.vertcat(np.eye(self.nx+self.nu), -np.eye(self.nx+self.nu))
        x_max = np.array([5., 5, 2*np.pi, 0.4, 1.]) # x, y, theta, v, w
        x_min = np.array([-5, -5, -2*np.pi, -0.2, -1.])
        u_max = np.array([0.1, 0.2]) # linear accel, angular accel
        u_min = np.array([-0.1, -0.2])
        x_max_f = np.array([0.5, 0.5, 2*np.pi, 0.5]) # x, y, theta, v
        x_min_f = np.array([-0.5, -0.5, -2*np.pi, -0.5])

        self.g = np.concatenate((x_max, u_max, -x_min, -u_min))
        self.ni = 2*(self.nx+self.nu)
        self.Gf = ca.vertcat(np.eye(self.nx), -np.eye(self.nx))
        # self.gf = np.concatenate((x_max_f, -x_min_f))
        self.gf = np.concatenate((x_max, -x_min))
        self.ni_f = 2*self.nx
        
        self.E_init = np.diag([0.025, 0.025, 0.05, 1e-4, 1e-4])
        self.E = np.diag([0.01, 0.01, 0.01, 1e-4, 1e-4])
        
        self.nw = self.nx

    def ode(self, X, u, w):
        x, y, theta, v_lin, v_ang = ca.vertsplit(X)
        xdot = v_lin * np.cos(theta)
        ydot = v_lin * np.sin(theta)
        thetadot = v_ang
        vdot = u[0]
        wdot = u[1]
        Xd = ca.vertcat(xdot, ydot, thetadot, vdot, wdot) + self.E @ w
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
        ax.plot(X[0, :], X[1, :], 'g-', linewidth=2.5, label='')

        # plt.legend()
        
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