import numpy as np
from dyn.model import Model
import casadi as ca
import os

base_path = os.path.dirname(__file__)

class G1(Model):

    def __init__(self, dt=0.01):
        self.nq = 30
        self.ndq = 29
        self.nx = self.nq + self.ndq

        self.nu = self.ndq - 6
        self.dt = dt    
        self.G = ca.vertcat(np.eye(self.nx + self.nu), -np.eye(self.nx + self.nu))
        q_min = np.array([-10., -10., -10., -1., -1., -1., -1., 
                          -2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618, # left: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
                          -2.5307, -2.9671, -2.7576, -0.087267, -0.87267, -0.2618, # right: 
                          -2.618, # waist_yaw
                          -3.0892, -1.5882, -2.618, -1.0472, -1.97222205, # left: shoulder_pitch, shoulder_roll, shoulder_yaw, left_elbow, left_wirst
                          -3.0892, -2.2515, -2.618, -1.0472, -1.97222205  # right: 
                          ])
        q_max = np.array([10., 10., 10., 1., 1., 1., 1.,
                          2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618,
                          2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618,
                          2.618, 
                          2.6704, 2.2515, 2.618, 2.0944, 1.97222205,
                          2.6704, 1.5882, 2.618, 2.0944, 1.97222205
                          ]) 

        lin_vel_lb = -30.0 * np.ones(3)
        lin_vel_ub = 30.0 * np.ones(3)
        ang_vel_lb = -np.pi * 2 * 20 * np.ones(3)
        ang_vel_ub = np.pi * 2 * 20 * np.ones(3)
        joint_vel_limit = np.pi * 2 * 10 * np.ones(self.ndq - 6)
        joint_vel_lb = -joint_vel_limit
        joint_vel_ub = joint_vel_limit
        v_max = np.concatenate([lin_vel_ub, ang_vel_ub, joint_vel_ub])
        v_min = np.concatenate([lin_vel_lb, ang_vel_lb, joint_vel_lb])

        x_max = np.concatenate([q_max, v_max])
        x_min = np.concatenate([q_min, v_min])
        u_max = 500 * np.ones(self.nu)
        u_min = -u_max

        self.g = np.concatenate((x_max, u_max, -x_min, -u_min))
        self.ni = 2 * (self.nx + self.nu)
        self.Gf = ca.vertcat(np.eye(self.nx), -np.eye(self.nx))
        self.gf = np.concatenate((x_max, -x_min))
        self.ni_f = 2 * self.nx
        # self.E = 1e-3 * np.eye(self.nx) 
        # self.E = 0.1 * np.eye(self.nx) 
        self.E = 5e-2 * np.eye(self.nx) 
        self.nw = self.nx
        
        self.ddyn_fun = ca.external("G1_ddyn", os.path.join(base_path, "G1_ddyn.so"), 
                                    {"enable_forward": True, "enable_reverse": True})

    def ddyn(self, x, u, w=None, h=0.05):
        if w is None:
            if isinstance(x, np.ndarray):
                w = np.zeros((self.nw))
            elif isinstance(x, ca.MX):
                w = ca.DM.zeros(self.nw, 1)

        return self.ddyn_fun(x, u) + self.dt * self.E @ w