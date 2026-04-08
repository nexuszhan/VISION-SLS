from perception_utils.networks import *
import numpy as np
from pathlib import Path
import yaml
from dotmap import DotMap
from pydrake.all import Variables, MathematicalProgram
import pydrake.symbolic as sym
from perception_utils.symbolic_utils import sym_to_pytorch
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client as bc
import os
from pybullet_envs.bullet import racecar
from perception_utils.eval_dino import *
# from torchvision import transforms
import shutil
import time
from matplotlib import pyplot as plt
import matplotlib.patches as patches

base_path = os.path.dirname(__file__)
load_path = os.path.join(base_path, 'data/learn_model_error_observability.yaml')
conf = yaml.safe_load(Path(load_path).read_text())
params = DotMap(conf)
device = 'cuda' if torch.cuda.is_available() else "cpu"
dino_feature_reduction = SupervisedDinoObservabilityLargeUnicycle(params.model).to(device) # p in paper
if params.train.load_ckpt_bound is not None:
    checkpoint = torch.load(os.path.join(base_path, "data/unicycle_checkpoint_observability_ft_075w_3pi4_097_300000_post.pt"), map_location=device)
    dino_feature_reduction.load_state_dict(checkpoint)

# Setting up Pybullet
def make_box(env, half_extents, position, euler_angles, lateral_friction=0.7, color=[0.02, 0.02, 0.02, 1.]):
    col_id = env.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    vis_id = env.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    obj_id = env.createMultiBody(0, col_id, vis_id, basePosition=position,
                               baseOrientation=p.getQuaternionFromEuler(euler_angles))
    env.changeDynamics(obj_id, -1, lateralFriction=lateral_friction)
    return obj_id

def set_state(env, robot, x, pos_orig, orn_orig_eul):
    position = list(pos_orig)
    position[:2] = x[:2] # x and y translations
    orientation = orn_orig_eul
    orientation[-1] = x[2]
    env.resetBasePositionAndOrientation(robot, position, p.getQuaternionFromEuler(orientation))

def settle(env, T=1000):
    for i in range(T):
        env.stepSimulation()

distance = 100000
img_w, img_h = 64, 64
nearVal, farVal, fov, aspect = 0.02, 25.5, 90, 1.
# It's necessary to enable GUI in the experiment
env = bc.BulletClient(connection_mode=p.GUI) 

env.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
env.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
env.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
env.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

env.resetSimulation()
env.setTimeStep(0.001)
urdfRoot = pybullet_data.getDataPath()
env.loadURDF(os.path.join(base_path, "plane_v2/plane_v2.urdf"))
env.setGravity(0, 0, -10)
robot_env = racecar.Racecar(env, urdfRootPath=urdfRoot, timeStep=0.001) 
robot = robot_env.racecarUniqueId
b_id_w1 = make_box(env, [1., 0.75, 0.01], [1.75, 0, 0.55], [0., 0., 0.], color=[0.2, 0.2, 0.2, 0.97])
settle(env)
out1, out2 = env.getBasePositionAndOrientation(robot)
view_matrix = env.computeViewMatrix(
            cameraEyePosition=[1.6, 0, 2.25],
            cameraTargetPosition=[1.6, 0, 0],
            cameraUpVector=[0, 1., 0.0]
        )
projection_matrix = env.computeProjectionMatrixFOV(
            fov=fov, aspect=aspect, nearVal=nearVal, farVal=farVal)

# Setting up Dino
preprocess = get_dino_v2_preprocess()
tf = transforms.ToPILImage()
model_dino = load_dino_v2_model("vitg14").to(device) 
print("dino loaded")
model_dino.patch_embed.forward = patch_embed_forward.__get__(model_dino.patch_embed)
def extract_dino_v2_features_single(images_pil) -> torch.Tensor:
    images = torch.stack([preprocess(tf(image.permute(2, 0, 1))) for image in images_pil], dim=0).to(device)
    with torch.no_grad():
        descriptor = model_dino.forward_features(images)["x_norm_clstoken"].detach().cpu()

    return descriptor
def dino_fn(x):
    # Set Pybullet states
    set_state(env, robot, x, list(out1), list(p.getEulerFromQuaternion(out2)))
    x_dim, y_dim, rgb_im, depth_im, seg_im = env.getCameraImage(8 * img_w, 8 * img_h,
                                                             view_matrix,
                                                             projection_matrix, shadow=True,
                                                             renderer=p.ER_BULLET_HARDWARE_OPENGL)
    # Load Dino model and set up transforms
    rgb_im = np.array(rgb_im, dtype=np.uint8).reshape((x_dim, y_dim, 4))
    descriptor = extract_dino_v2_features_single(torch.from_numpy(np.array([rgb_im[:,:,:-1]])).to(device))
    nn_descriptor = dcn(dino_feature_reduction(descriptor.to(device)))
    return nn_descriptor, rgb_im[:,:,:-1]

from dyn.unicycle_of_perception import Unicycle_OF
N = 30
m = Unicycle_OF()
m.dt = 0.15
x_max = np.array([0.5, 1.9, 2*np.pi, 2.]) # x, y, theta, v
x_min = np.array([-3.5, -2., -2*np.pi, -1.])
u_max = np.array([np.pi, 4.]) # angular_vel, linear acceleration
u_min = np.array([-np.pi, -4.])
m.g = np.concatenate((x_max, u_max, -x_min, -u_min))
x_max_f0 = np.array([0.45, 0.65, 2*np.pi, 1.]) # x, y, theta, v
x_min_f0 = np.array([0., 0.35, -2*np.pi, -1.])
x_max_f1 = np.array([0.2, 0.2, 2*np.pi, 1.]) # 1-6, 14-
x_min_f1 = np.array([-0.2, -0.2, -2*np.pi, -1.])
x_max_f2 = np.array([0.2, 1.7, 2*np.pi, 1.]) # 7-10
x_min_f2 = np.array([-0.2, 1.3, -2*np.pi, -1.])
x_max_f3 = np.array([0.2, -1.05, 2*np.pi, 1.]) # 11-13
x_min_f3 = np.array([-0.2, -1.45, -2*np.pi, -1.])

obstacles = [(np.array([-3.55, 0.7]), 0.4), (np.array([-0.6, 1.15]), 0.35), (np.array([-1.2, -1.2]), 0.4)]
obstacle_centers = np.stack([obs[0] for obs in obstacles], axis=0)
obstacle_r2 = np.array([obs[1] * obs[1] for obs in obstacles])

x_min_f_all = [x_min_f0] + [x_min_f1]*6 + [x_min_f2]*4 + [x_min_f3]*3 + [x_min_f1]
x_max_f_all = [x_max_f0] + [x_max_f1]*6 + [x_max_f2]*4 + [x_max_f3]*3 + [x_max_f1]
x0_all = [np.array([-2.1, -1.75, np.pi/2, 0.]), np.array([-2.5, -1.5, 0., 0.]), np.array([-2.5, -1., 0., 0.]),
          np.array([-2., 1.5, 0., 0.]), np.array([-1.25, 1.2, 0., 0.]), np.array([-0.75, 1.75, -np.pi/2, 0.]),
          np.array([-1.5, -1.75, np.pi/2, 0.]), np.array([-2.5, 0., np.pi*0.35, 0.]), np.array([-2.5, 0.5, np.pi/4, 0.]),
          np.array([-2.5, 1., 0., 0.]), np.array([-2.75, 1.25, np.pi/4, 0.]), np.array([-2., -1.5, 0., 0.]),
          np.array([-2.5, -1.5, 0., 0.]), np.array([-2.25, -0.75, 0., 0.]), np.array([-2.2, -0.6, 0.6, 0.])]

state_offset = np.array([3., 0., 0., 0.])

eval_idx = 0
# data = np.load(os.path.join(base_path, "data/unicycle_of_perception.npz")) 
# data = np.load(os.path.join(base_path, "data/unicycle_of_perception_unrobust.npz")) 
# data = np.load(os.path.join(base_path, "data/unicycle_of_perception_no_info_gather.npz")) 
x0 = x0_all[eval_idx]
x_min_f = x_min_f_all[eval_idx]
x_max_f = x_max_f_all[eval_idx]
m.gf = np.concatenate((x_max_f, -x_min_f))
data = np.load(os.path.join(base_path, f"data/unicycle_of_perception_{eval_idx}.npz"))
# data = np.load(os.path.join(base_path, f"data/unicycle_of_perception_unrobust_{eval_idx}.npz"))
# data = np.load(os.path.join(base_path, f"data/unicycle_of_perception_CE_{eval_idx}.npz"))
primal_x = data["nominal_traj"]
primal_u = data["nominal_input"]
Phi_xx = data["Phi_xx"]
Phi_ux = data["Phi_ux"]
Phi_xy = data["Phi_xy"]
Phi_uy = data["Phi_uy"]
tube = data["backoff"]
tube_f = data["backoff_f"]
K_mat = Phi_uy - Phi_ux @ np.linalg.inv(Phi_xx) @ Phi_xy

ROLLOUT_NUM = 50
rollout_states = np.zeros((ROLLOUT_NUM, m.nx, N+1))
rollout_states_open = np.zeros((ROLLOUT_NUM, m.nx, N+1))
feedback_error = 0.
feedback_error_terminal = 0.
openloop_error = 0.
openloop_error_terminal = 0.
np.random.seed(42)

success = 0
constr_violation = 0

for rollout in range(ROLLOUT_NUM):
    print("rollout"+str(rollout))
    states = np.zeros((m.nx, N+1))
    states[:,0] = x0
    observations = np.zeros((m.ny, N+1))
    disturbance_state = np.zeros((m.nx, N))
    disturbance_obs = np.zeros((m.ny, N))
    
    if rollout == 0: # some adversarial disturbance
        disturbance_state[0,:] = -1.
    elif rollout == 1:
        disturbance_state[1,:] = -1.
    elif rollout == 2:
        disturbance_state[2,:] = -1.
    elif rollout == 3:
        disturbance_state[3,:] = -1.
    elif rollout == 4:
        disturbance_state[0,:] = 1.
    elif rollout == 5:
        disturbance_state[1,:] = 1.
    elif rollout == 6:
        disturbance_state[2,:] = 1.
    elif rollout == 7:
        disturbance_state[3,:] = 1.
    else:
        for i in range(N):
            disturbance_state[:,i] = (np.random.random(m.nx)-0.5) 
            disturbance_state[:,i] = disturbance_state[:,i] / np.linalg.norm(disturbance_state[:,i])
    
    for t in range(1, N+1):
        delta_u = np.zeros((m.nu))
        # TODO: add a x(-1) to compute y(0) in this case
        if t == 1:
            obs, rgb_im = dino_fn(states[:,t-1] + state_offset)
            observations[:,t-1] = obs[0]
        else:
            obs, rgb_im = dino_fn(states[:,t-2] + state_offset)
            observations[:,t-1] = obs[0]
        
        delta_u += K_mat[(t-1)*m.nu:t*m.nu, 0:m.ny] @ (observations[:,0] - m.C @ (primal_x[:,0] + state_offset))
        for j in range(1, t):
            delta_u += K_mat[(t-1)*m.nu:t*m.nu, j*m.ny:(j+1)*m.ny] @ (observations[:,j] - m.C @ (primal_x[:,j-1] + state_offset))
        
        u = primal_u[:,t-1] + delta_u

        states[:,t] = np.squeeze(m.ddyn(states[:,t-1], u, disturbance_state[:,t-1], m.dt))

        feedback_error += np.linalg.norm(states[:,t] - primal_x[:,t])
        # plt.imsave(f"video_frames_nonrobust2/frame_{t-1:03d}.png", rgb_im)
    rollout_states[rollout, :, :] = states.copy()
    feedback_error_terminal += np.linalg.norm(states[:,-1] - primal_x[:,-1])
    
    path_states = states[:, :-1]
    final_state = states[:, -1]
    path_in_bounds = np.all((path_states >= x_min[:, None]-1e-3) & (path_states <= x_max[:, None]+1e-3))
    final_in_bounds = np.all((final_state >= x_min_f-1e-3) & (final_state <= x_max_f+1e-3))
    arrive = np.all((final_state[:2] >= x_min_f[:2]-1e-3) & (final_state[:2] <= x_max_f[:2]+1e-3))

    xy_states = states[:2, :].T
    delta_obs = xy_states[:, None, :] - obstacle_centers[None, :, :]
    collides = np.any(np.sum(delta_obs * delta_obs, axis=2) <= obstacle_r2[None, :]+1e-4)

    success += int(arrive and not collides)
    constr_violation += int((not path_in_bounds) or (not final_in_bounds) or collides)

    # disturbance_state_open = disturbance_state.copy()  

    # states_open = np.zeros((m.nx, N + 1))
    # states_open[:, 0] = x0
    # for t in range(1, N + 1):
    #     u_ol = primal_u[:, t - 1]  # scheduled input only
    #     states_open[:, t] = np.squeeze(
    #         m.ddyn(states_open[:, t - 1], u_ol, disturbance_state_open[:, t - 1], m.dt)
    #     )

    #     openloop_error += np.linalg.norm(states_open[:,t] - primal_x[:,t])
    # openloop_error_terminal += np.linalg.norm(states_open[:,-1] - primal_x[:,-1])
    # rollout_states_open[rollout, ...] = states_open.copy()

np.savez(f"data/rollout_{eval_idx}.npz", 
         rollout_states=rollout_states,
         rollout_states_open=rollout_states_open,
         feedback_error=feedback_error,
         feedback_error_terminal=feedback_error_terminal,
         openloop_error=openloop_error,
         openloop_error_terminal=openloop_error_terminal
         )

env.disconnect()

print(f"test {eval_idx}")
print("success rate: {:.2f}%".format(success/ROLLOUT_NUM * 100))
print("constraint violation rate: {:.2f}%".format(constr_violation/ROLLOUT_NUM * 100))
