from dyn.model import Model
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


class LightDark(Model):
    def __init__(self):
        self.nx = 2
        self.nu = 2
        self.dt = 0.2

        self.G = ca.vertcat(np.eye(self.nx+self.nu), -np.eye(self.nx+self.nu))
        x_max = np.array([5., 5.])
        x_min = np.array([-2., -2.])
        self.x_f_max = np.array([0.15, 0.15])
        self.x_f_min = np.array([-0.15, -0.15])
        u_max = np.array([1, 1])

        self.g = np.concatenate((x_max, u_max, -x_min, u_max))
        self.ni = 2*(self.nx+self.nu)
        self.Gf = ca.vertcat(np.eye(self.nx), -np.eye(self.nx))
        self.gf = np.concatenate((self.x_f_max, -self.x_f_min))
        self.ni_f = 2*self.nx

        self.E_init = 0.05 * np.eye(self.nx) # error on the initial condition
        self.nw = self.nx

    def ode(self, X, u, w):
        x1, x2 = ca.vertsplit(X)

        return ca.vertcat(u[0]+0.05*w[0], u[1]+0.05*w[1])

    def replace_constraints(self, x_max, x_min, u_max, u_min, x_max_f, x_min_f):
        self.g = np.hstack((x_max, u_max, -x_min, -u_min))
        self.gf = np.hstack((x_max_f, -x_min_f))
    
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
        
        from matplotlib.patches import Rectangle
        x_min_f = self.x_f_min
        x_max_f = self.x_f_max
        ax.add_patch(Rectangle(
            (x_min_f[0], x_min_f[1]),
            x_max_f[0] - x_min_f[0],
            x_max_f[1] - x_min_f[1],
            fill=False,
            edgecolor='k',
            linewidth=1.,
            label='Terminal constraint'
        ))

        # Plot the nominal trajectory
        ax.plot(X[0, :], X[1, :], 'r-', linewidth=2, label='Nominal trajectory')
        
        # Plot the initial condition
        ax.plot(X[0, 0], X[1, 0], 'bo', markersize=8, label='Initial state')
        
        # Plot the final condition
        ax.plot(X[0, -1], X[1, -1], 'go', markersize=8, label='Final state')

        plt.legend()
        
        return ax

    @staticmethod
    def get_tubes_x(Phi_x, density=100):
        """
        Get tubes for state variables using ellipsoid approximation.
        
        Args:
            Phi_x: Phi_x matrix from SLS solution
            density: number of points for ellipsoid approximation
        
        Returns:
            List of tubes (polygons)
        """
        from util.minkowski_sum_ellipsoids import minkowski_sum_ellipsoids
        
        tubes = []
        for i in range(Phi_x.shape[1]):
            tubes += [minkowski_sum_ellipsoids(Phi_x[i], num_points=density, resample_points=density)]
        return tubes

    @staticmethod
    def plot_tube(tube, ax=None):
        """
        Plot a single tube (ellipsoid) on the given axis.
        
        Args:
            tube: tube points (polygon vertices)
            ax: matplotlib axis (optional)
        
        Returns:
            matplotlib axis with the tube plot
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MatplotlibPolygon
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        polygon = MatplotlibPolygon(
            tube,
            edgecolor='none',  # No edge color
            facecolor='blue',  # Filled blue background
            alpha=0.3  # Transparency of the background
        )
        
        ax.add_patch(polygon)
        
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
