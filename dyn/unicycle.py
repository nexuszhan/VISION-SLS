import numpy as np
from dyn.model import Model
import casadi as ca
import matplotlib.pyplot as plt

class Unicycle(Model):
    def __init__(self):
        self.nx = 4
        self.nu = 2
        self.dt = 0.2

        self.G = ca.vertcat(np.eye(self.nx+self.nu), -np.eye(self.nx+self.nu))
        x_max = np.array([5, 5, 2*np.pi, 2.]) # x, y, theta, v
        x_min = np.array([-5, -5, -2*np.pi, -1.])
        u_max = np.array([np.pi/4, 1.]) # angular_vel, linear acceleration
        u_min = np.array([-np.pi/4, -1.])
        x_max_f = np.array([0.5, 0.5, 2*np.pi, 0.5]) # x, y, theta, v
        x_min_f = np.array([-0.5, -0.5, -2*np.pi, -0.5])

        self.g = np.concatenate((x_max, u_max, -x_min, -u_min))
        self.ni = 2*(self.nx+self.nu)
        self.Gf = ca.vertcat(np.eye(self.nx), -np.eye(self.nx))
        # self.gf = np.concatenate((x_max_f, -x_min_f))
        self.gf = np.concatenate((x_max, -x_min))
        self.ni_f = 2*self.nx
        
        self.E = 0.01 * np.eye(self.nx)
        
        self.nw = self.nx

    def ode(self, X, u, w):
        x, y , theta, v = ca.vertsplit(X)
        xdot = v * np.cos(theta)
        ydot = v * np.sin(theta)
        thetadot = u[0]
        vdot = u[1]
        Xd = ca.vertcat(xdot, ydot, thetadot, vdot) + self.E @ w
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
        ax.plot(X[0, :], X[1, :], 'g-', linewidth=2.5, label="nominal trajectory")

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