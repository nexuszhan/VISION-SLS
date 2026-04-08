from perception_utils.networks import *
import numpy as np
from pathlib import Path
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client as bc
import os
from perception_utils.eval_dino import *
# from torchvision import transforms
import shutil
import time
from matplotlib import pyplot as plt
import matplotlib.patches as patches

base_path = os.path.dirname(__file__)
device = 'cpu' 

def set_state(env, robot, x, joint_mapping):
    base_pos = x[:3]
    base_pos[2] -= 0.4
    base_quat = x[3:7] / np.linalg.norm(x[3:7])
    env.resetBasePositionAndOrientation(robot, base_pos, base_quat)

    num_joints = env.getNumJoints(robot)
    # for i in range(num_joints):
    # for i in range(30):
        # p.resetJointState(robot, i, x[7 + i])
    for i in range(len(joint_mapping)):
        env.resetJointState(robot, joint_mapping[i], x[i+7])

def settle(env, T=1000):
    for i in range(T):
        env.stepSimulation()

distance = 100000
img_w, img_h = 64, 64
nearVal, farVal, fov, aspect = 0.02, 25.5, 90, 1.
env = bc.BulletClient(connection_mode=p.GUI)

env.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
env.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
env.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
env.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

env.resetSimulation()
env.setTimeStep(1/240)
urdfRoot = pybullet_data.getDataPath()
plane = env.loadURDF(os.path.join(urdfRoot, "plane.urdf"))
env.setGravity(0, 0, 0)
repo_root = Path(__file__).resolve().parents[2]
robot = env.loadURDF(os.path.join(repo_root, "dyn/G1/g1_23dof.urdf"), globalScaling=0.5)

joint_mapping = np.array([1,2,3,4,5,6, 
                          7,8,9,10,11,12,
                          13,
                          21,22,23,24,25,
                          26,27,28,29,30])

settle(env)

from scipy.spatial.transform import Rotation as R
from dyn.G1.G1_of import G1OF
m = G1OF()
x_min = -m.g[m.nx+m.nu:m.nx*2+m.nu]
x_max = m.g[:m.nx]
x_max_f = x_max.copy()
x_min_f = x_min.copy()

data = np.load(os.path.join(base_path, "data/G1_robust.npz")) 
primal_x = data["nominal_traj"]
primal_u = data["nominal_input"]
Phi_xx = data["Phi_xx"]
Phi_ux = data["Phi_ux"]
Phi_xy = data["Phi_xy"]
Phi_uy = data["Phi_uy"]
tube = data["backoff"]
tube_f = data["backoff_f"]

x0 = primal_x[:,0]
N = primal_x.shape[1] - 1 

set_state(env, robot, x0[:m.nq].copy(), joint_mapping)
out1, out2 = env.getBasePositionAndOrientation(robot)
rot90 = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
view_matrix = env.computeViewMatrix(
            cameraEyePosition=list(np.array(out1) + np.array([1., 0., 0.5])),
            cameraTargetPosition=list(out1),
            cameraUpVector=[0, 0, 1.0]
        )
projection_matrix = env.computeProjectionMatrixFOV(
            fov=fov, aspect=aspect, nearVal=nearVal, farVal=farVal)

time.sleep(1.)

K_mat = Phi_uy - Phi_ux @ np.linalg.inv(Phi_xx) @ Phi_xy
side_view = False
ROLLOUT_NUM = 50
np.random.seed(0)
rollout_states = []
rollout_states_open = []
success = 0
constr_violation = 0
for rollout in range(ROLLOUT_NUM):
    print("rollout"+str(rollout))
    states = np.zeros((m.nx, N+1))
    states[:,0] = x0
    observations = np.zeros((m.ny, N+1))
    disturbance_state = np.zeros((m.nx, N))
    disturbance_obs = np.zeros((m.ny, N))

    for i in range(N):
        disturbance_state[rollout,i] = -1.
    for i in range(N):
        disturbance_obs[:,i] = (np.random.random(m.ny)-0.5) 
        disturbance_obs[:,i] = disturbance_obs[:,i]/np.linalg.norm(disturbance_obs[:,i])
    
    
    for t in range(1, N+1):
        camera_pos = np.array(out1) + np.array([1., 0., 0.5])
        if side_view:
            camera_pos = rot90 @ camera_pos
        view_matrix = env.computeViewMatrix(
                    cameraEyePosition=list(camera_pos),
                    cameraTargetPosition=list(out1),
                    cameraUpVector=[0, 0, 1.0]
                )
        projection_matrix = env.computeProjectionMatrixFOV(
                    fov=fov, aspect=aspect, nearVal=nearVal, farVal=farVal)
        
        delta_u = np.zeros((m.nu))
        # TODO: add a x(-1) to compute y(0) in this case
        if t == 1:
            observations[:,t-1] = m.measurement(states[:,t-1], disturbance_obs[:,t-1])
            set_state(env, robot, states[:,t-1].copy(), joint_mapping)
            x_dim, y_dim, rgb_im, depth_im, seg_im = env.getCameraImage(8 * img_w, 8 * img_h,
                                                        view_matrix,
                                                        projection_matrix, shadow=True,
                                                        renderer=p.ER_BULLET_HARDWARE_OPENGL)
 
            rgb_im = np.array(rgb_im, dtype=np.uint8).reshape((x_dim, y_dim, 4))
            rgb_im = rgb_im[:,:,:-1]
        else:
            observations[:,t-1] = m.measurement(states[:,t-2], disturbance_obs[:,t-1])
            set_state(env, robot, states[:,t-2].copy(), joint_mapping)
            x_dim, y_dim, rgb_im, depth_im, seg_im = env.getCameraImage(8 * img_w, 8 * img_h,
                                                        view_matrix,
                                                        projection_matrix, shadow=True,
                                                        renderer=p.ER_BULLET_HARDWARE_OPENGL)
            rgb_im = np.array(rgb_im, dtype=np.uint8).reshape((x_dim, y_dim, 4))
            rgb_im = rgb_im[:,:,:-1]
        
        delta_u += K_mat[(t-1)*m.nu:t*m.nu, 0:m.ny] @ (observations[:,0] - m.C @ (primal_x[:,0]))
        for j in range(1, t):
            delta_u += K_mat[(t-1)*m.nu:t*m.nu, j*m.ny:(j+1)*m.ny] @ (observations[:,j] - m.C @ (primal_x[:,j-1]))
        
        u = primal_u[:,t-1] + delta_u

        states[:,t] = np.squeeze(m.ddyn(states[:,t-1], u, disturbance_state[:,t-1], m.dt))
        time.sleep(0.01)
        
        # check if all states lie in tubes
        if t < N:
            for ii in range(m.nx):
                if np.fabs(states[ii,t] - primal_x[ii,t]) > tube[t,ii]:
                    print("warning")
                
                if np.fabs(primal_x[ii,t]+tube[t,ii] - m.g[ii]) <= 1e-3:
                    print("active constraint {:} at time {:}".format(ii, t))
                if np.fabs(primal_x[ii,t]-tube[t,ii] + m.g[ii+m.nx+m.nu]) <= 1e-3:
                    print("active constraint {:} at time {:}".format(ii+m.nx+m.nu, t))
        else:
            for ii in range(m.nx):
                if np.fabs(states[ii,t] - primal_x[ii,t]) > tube_f[ii]:
                    print("warning")
                
                if np.fabs(primal_x[ii,t]+tube_f[ii] - m.gf[ii]) <= 1e-3:
                    print("active constraint {:} at time {:}".format(ii, t))
                if np.fabs(primal_x[ii,t]-tube_f[ii] + m.gf[ii+m.nx]) <= 1e-3:
                    print("active constraint {:} at time {:}".format(i+m.nx, t))
        
        if side_view:
            plt.imsave(f"video_frames/frame_{t-1:03d}_side.png", rgb_im)
        else:
            plt.imsave(f"video_frames/frame_{t-1:03d}.png", rgb_im)
    time.sleep(1.)
    
    disturbance_state_open = disturbance_state.copy()  # NEW

    states_open = np.zeros((m.nx, N + 1))
    states_open[:, 0] = x0
    for t in range(1, N + 1):
        u_ol = primal_u[:, t - 1]  # scheduled input only
        states_open[:, t] = np.squeeze(
            m.ddyn(states_open[:, t - 1], u_ol, disturbance_state_open[:, t - 1], m.dt)
        )

        # set_state(env, robot, states_open[:,t-1].copy(), joint_mapping)
        # x_dim, y_dim, rgb_im, depth_im, seg_im = env.getCameraImage(8 * img_w, 8 * img_h,
        #                                             view_matrix,
        #                                             projection_matrix, shadow=True,
        #                                             renderer=p.ER_BULLET_HARDWARE_OPENGL)
        # # Load Dino model and set up transforms
        # rgb_im = np.array(rgb_im, dtype=np.uint8).reshape((x_dim, y_dim, 4))
        # rgb_im = rgb_im[:,:,:-1]
        # if side_view:
        #     plt.imsave(f"video_frames_open/frame_{t-1:03d}_side.png", rgb_im)
        # else:
        #     plt.imsave(f"video_frames_open/frame_{t-1:03d}.png", rgb_im)

    # if t < N:
    #     for ii in range(m.nx):
    #         if np.fabs(states_open[ii,t] - primal_x[ii,t]) > tube[t,ii]:
    #             print("warning: ", ii)
    # else:
    #     for ii in range(m.nx):
    #         if np.fabs(states_open[ii,t] - primal_x[ii,t]) > tube_f[ii]:
    #             print("warning: ", ii)

    path_states = states_open[:, :-1]
    final_state = states_open[:, -1]
    path_in_bounds = np.all((path_states >= x_min[:, None]-1e-3) & (path_states <= x_max[:, None]+1e-3))
    final_in_bounds = np.all((final_state >= x_min_f-1e-3) & (final_state <= x_max_f+1e-3))
    arrive = np.all((final_state[:7] >= x_min_f[:7]-1e-3) & (final_state[:7] <= x_max_f[:7]+1e-3))

    success += int(arrive)
    constr_violation += int((not path_in_bounds) or (not final_in_bounds))


    rollout_states.append(states.copy())
    rollout_states_open.append(states_open.copy())

np.savez("data/G1_rollout.npz", 
         rollout_states = np.array(rollout_states),
         rollout_states_open = np.array(rollout_states_open))

env.disconnect()

print("success rate: {:.2f}%".format(success/ROLLOUT_NUM * 100))
print("constraint violation rate: {:.2f}%".format(constr_violation/ROLLOUT_NUM * 100))