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
load_path = os.path.join(base_path, "data/learn_model_error_observability.yaml")
conf = yaml.safe_load(Path(load_path).read_text())
params = DotMap(conf)
device = 'cuda' if torch.cuda.is_available() else "cpu"
dino_feature_reduction = SupervisedDinoObservabilityLarge(params.model).to(device) # p in paper
checkpoint = torch.load(os.path.join(base_path, "data/quad10d_checkpoint_observability_ft_2000000_snowdensev4.pt"), map_location=device)
checkpoint_clean = checkpoint["C"]
checkpoint_clean[checkpoint_clean.abs() <= 5e-2] = 0.
dino_feature_reduction.load_state_dict(checkpoint)

# Setting up Pybullet
def make_cylinder(env, radius, height, position, euler_angles, mass=1., lateral_friction=1.5, spinning_friction=0.1, color=[0.8, 0.7, 0.3, 0.8]):
    col_id = env.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
    vis_id = env.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
    obj_id = env.createMultiBody(mass, col_id, vis_id, basePosition=position,
                               baseOrientation=p.getQuaternionFromEuler(euler_angles))
    env.changeDynamics(obj_id, -1, lateralFriction=lateral_friction, spinningFriction=spinning_friction)
    return obj_id

def set_state(env, robot, x, pos_orig):
    position = list(pos_orig)
    position[0] = x[0] # x translation
    position[1] = x[1]  # z translation
    position[2] = x[2]  # z translation

    orientation = [0., 0., 0.]
    orientation[1] = x[3]
    orientation[2] = x[4]
    env.resetBasePositionAndOrientation(robot, position, p.getQuaternionFromEuler(orientation))

def settle(env, T=1000):
    for i in range(T):
        env.stepSimulation()

img_w, img_h = 64, 64  # 80, 80
nearVal = 0.02
farVal = 15.5  # 3.5
fov = 90
aspect = 1.
distance = 100000
env = bc.BulletClient(connection_mode=p.GUI)
timeStep = 0.001

env.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
env.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
env.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
env.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

env.resetSimulation()
env.setTimeStep(timeStep)
urdfRoot = pybullet_data.getDataPath()
robot = env.loadURDF(os.path.join(base_path, "cf2p/cf2p.urdf"), globalScaling=0.04)
env.loadURDF(os.path.join(base_path, "plane_v4_snow/plane_v4_snow.urdf"))
env.loadURDF(os.path.join(base_path, "wall/snow_left.urdf"))
env.loadURDF(os.path.join(base_path, "wall/snow_right.urdf"))
env.loadURDF(os.path.join(base_path, "wall/snow_center.urdf"))
env.setGravity(0, 0, -10)
c1 = make_cylinder(env, 0.1, 2., [1., 1., 1.], [0, 0., 0.], mass=0., color=[0.5, 0.0, 0.0, 1])
c2 = make_cylinder(env, 0.1, 2.5, [-1.5, 4., 1.25], [0, 0., 0.], mass=0., color=[0.0, 0.5, 0.0, 1])
c3 = make_cylinder(env, 0.1, 3., [0.5, 3., 1.5], [0, 0., 0.], mass=0., color=[0.0, 0.0, 0.5, 1])
c4 = make_cylinder(env, 0.1, 1., [-0.5, 1., 0.5], [0, 0., 0.], mass=0., color=[0.8, 0.7, 0.3, 1])
c5 = make_cylinder(env, 0.1, 1.5, [-1, 2.5, 0.75], [0, 0., 0.], mass=0., color=[0.0, 0.7, 0.7, 1])

c6 = make_cylinder(env, 0.1, 2.25, [2.25, 4.5, 1.125], [0, 0., 0.], mass=0., color=[0.75, 0.2, 0.0, 1])
c7 = make_cylinder(env, 0.1, 2.75, [0.25, 4.5, 1.375], [0, 0., 0.], mass=0., color=[0.25, 0.75, 0.0, 1])
c8 = make_cylinder(env, 0.1, 3.25, [2.3, 3., 1.625], [0, 0., 0.], mass=0., color=[0.75, 0.75, 0.75, 1])
c9 = make_cylinder(env, 0.1, 3.25, [-2.3, 0.5, 1.625], [0, 0., 0.], mass=0., color=[0.2, 0.0, 0.0, 1])
c10 = make_cylinder(env, 0.1, 1.75, [1.5, 4.5, 0.875], [0, 0., 0.], mass=0., color=[0.2, 0.3, 0.7, 1])

c11 = make_cylinder(env, 0.1, 3.5, [-2.25, 1.5, 1.75], [0, 0., 0.], mass=0., color=[1, 0.5, 0, 1])
c12 = make_cylinder(env, 0.1, 3.5, [-2.25, 3.5, 1.75], [0, 0., 0.], mass=0., color=[0.5, 0.1, 0.7, 1])
c13 = make_cylinder(env, 0.1, 3.5, [2.25, 1.5, 1.75], [0, 0., 0.], mass=0., color=[0., 1, 0.3, 1])
c14 = make_cylinder(env, 0.1, 3.5, [2.25, -0.5, 1.75], [0, 0., 0.], mass=0., color=[0.93, 0, 1, 1])
c15 = make_cylinder(env, 0.1, 3.5, [-2.25, -0.5, 1.75], [0, 0., 0.], mass=0., color=[0.1, 0.8, 0.0, 1])

c16 = make_cylinder(env, 0.1, 3.5, [-1, 4.5, 1.75], [0, 0., 0.], mass=0., color=[1, 0.35, 0, 1])
c17 = make_cylinder(env, 0.1, 3.5, [1, 4.5, 1.75], [0, 0., 0.], mass=0., color=[1, 1, 0.0, 1])
c18 = make_cylinder(env, 0.1, 3.5, [2.25, 0.5, 1.75], [0, 0., 0.], mass=0., color=[1, 0.35, 0.4, 1])
c19 = make_cylinder(env, 0.1, 3.5, [-2.25, 2.5, 1.75], [0, 0., 0.], mass=0., color=[1, 1, 1, 1])
c20 = make_cylinder(env, 0.1, 1.75, [-0.5, 4.5, 0.875], [0, 0., 0.], mass=0., color=[0., 0.2, 0., 1])

c21 = make_cylinder(env, 0.1, 3.5, [-2, -0.75, 1.75], [0, 0., 0.], mass=0., color=[1, 0.35, 0, 1])
c22 = make_cylinder(env, 0.1, 3.5, [-1, -0.75, 1.75], [0, 0., 0.], mass=0., color=[1, 0.8, 0.0, 1])
c23 = make_cylinder(env, 0.1, 3.5, [0, -0.75, 1.75], [0, 0., 0.], mass=0., color=[0.4, 0.35, 1, 1])
c24 = make_cylinder(env, 0.1, 3.5, [1, -0.75, 1.75], [0, 0., 0.], mass=0., color=[0.3, 0.3, 0.3, 1])
c25 = make_cylinder(env, 0.1, 1.75, [2, -0.75, 0.875], [0, 0., 0.], mass=0., color=[0.5, 0.2, 0., 1])
settle(env)
out1, out2 = env.getBasePositionAndOrientation(robot)

projection_matrix = env.computeProjectionMatrixFOV(
    fov=fov, aspect=aspect, nearVal=nearVal, farVal=farVal)
init_camera_vector = np.array([0, 1, 0])  # z-axis
init_camera_origin = np.array([0, 0., -0.])  # z-axis
init_up_vector = np.array([0, 0, 1])  # y-axis

# Setting up Dino
preprocess = get_dino_v2_preprocess()
tf = transforms.ToPILImage()
model_dino = load_dino_v2_model("vitl14").to(device)
model_dino.patch_embed.forward = patch_embed_forward.__get__(model_dino.patch_embed)
def extract_dino_v2_features_single(images_pil) -> torch.Tensor:
    images = torch.stack([preprocess(tf(image.permute(2, 0, 1))) for image in images_pil], dim=0).to(device)
    with torch.no_grad():
        descriptor = model_dino.forward_features(images)["x_norm_clstoken"].detach().cpu()

    return descriptor
def dino_fn(x):
    # Set Pybullet states
    x_pybullet = x[[0,1,2,6,7]]
    set_state(env, robot, x_pybullet, list(out1))
    pos, orn = env.getBasePositionAndOrientation(robot)
    rot_matrix = env.getMatrixFromQuaternion(orn)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    camera_vector = rot_matrix.dot(init_camera_vector)
    up_vector = rot_matrix.dot(init_up_vector)
    pos_shift = np.array(pos) + rot_matrix.dot(init_camera_origin)
    
    view_matrix = env.computeViewMatrix(pos_shift, pos_shift + 0.1 * camera_vector, up_vector)
    x_dim, y_dim, rgb_im, depth_im, seg_im = env.getCameraImage(8 * img_w, 8 * img_h,
                                                             view_matrix,
                                                             projection_matrix, shadow=True,
                                                             renderer=p.ER_BULLET_HARDWARE_OPENGL)
    # Load Dino model and set up transforms
    rgb_im = np.array(rgb_im, dtype=np.uint8).reshape((x_dim, y_dim, 4))
    descriptor = extract_dino_v2_features_single(torch.from_numpy(np.array([rgb_im[:,:,:-1]])).to(device))
    nn_descriptor = dcn(dino_feature_reduction(descriptor.to(device)))
    return nn_descriptor, rgb_im[:,:,:-1]

from dyn.quad_10d_of_perception import Quad10d_OF
m = Quad10d_OF()
m.dt = 0.15
x_max = m.g[:m.nx]
x_min = -m.g[m.nx+m.nu:m.nx*2+m.nu]
x_max_f = m.gf[:m.nx]
x_min_f = -m.gf[m.nx:]

x0_all = [
    np.array([0.9, 3.1, 1., 0.001, 0.001, 0.001, 0.001, -0.001, -0.001, -0.001]),
    np.array([-1.5, 1.5, 1., 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]),
    np.array([-1.5, 1., 1.4, 0.001, 0.001, 0.001, -0.001, -0.001, -0.001, -0.001]),
    np.array([-1.06593470, 1.5, 1.10456181, 2.67332447e-05, 3.83307696e-06, 
              7.08075505e-04, -7.57944103e-04, -6.61702048e-04, -3.65235395e-04, -6.55121415e-04]),
    np.array([1.10908462, 1.74590952, 1.23074318, 8.81476428e-04, -3.21073508e-04, 
              -2.24316359e-04, 3.01206361e-04, -9.99463201e-04, 3.67372965e-04, 8.55874892e-04]), 
    np.array([-1., 1.2, 1.43348860, -6.51474351e-04, -1.19435135e-05, 
              1.27502407e-04, 5.00949954e-04, -3.93577815e-04, -3.86666605e-04,  9.25646947e-04]),
    np.array([1.24921277, 3.30709532, 1.39855684, -3.54630626e-04, 9.60420333e-04, 
              -1.21189988e-04, 2.53206418e-04, 2.19917194e-04, -5.26161249e-06, -9.85192052e-04]),
    np.array([9.18497747e-01, 3.49854528, 1.43001580, -3.38818143e-04, 8.12694421e-04, 
              1.25370733e-04, 1.99184193e-04, 3.67443620e-05, -7.20104659e-04, 1.75409946e-04]),
    np.array([9.09025146e-01, 1.78647539, 1.01162208, 9.47650516e-04, -4.24438500e-04, 
              -3.27323338e-04, 1.52521899e-05, -9.33170630e-04, -8.26410964e-04, 1.50665656e-04]),
    np.array([1.38664083, 1.67795747, 1.33548519, 2.24796503e-04, 4.02040567e-04, 
              5.05786201e-04, -2.31956416e-05, 7.10199453e-04, -7.37747249e-04, -1.77197178e-04]),
    np.array([1.5, 1., 1.14684690, 8.68682345e-04, -1.83401133e-04, 
              -7.98845957e-04, -6.33092081e-04, 5.78230045e-05, -9.41358914e-07, 9.29638793e-04]),
    np.array([1.45267802, 1.19938593, 1.02210279, 9.47074739e-04, 7.81317063e-04, 
              7.47597111e-04, -5.36177642e-06, 2.49029350e-04, -2.07637014e-04, 7.84336561e-06]),
    np.array([1.43255604, 2.09522370, 1.17212090, 5.87481746e-05, 5.36093425e-04, 
              4.45792076e-04, -5.08720833e-05, 2.62424053e-04, -1.46258867e-04, 1.84814170e-04]),
    np.array([1.26462429, 2.70203864, 1.10651198, -3.24105474e-04, -9.03459809e-05, 
              2.85908747e-04, 1.31491574e-05, 5.16992626e-04, -3.39725758e-04, 7.16878499e-04]),
    np.array([1.01062306, 1.4, 1.41533893, -7.47929053e-04, 3.96140517e-04, 
              -3.42311859e-04, 8.37502304e-04, 5.13426549e-04, -4.52686696e-05, -2.36126558e-04]),
]
xG_all = [
    np.array([-1.5, 2.5, 1.4, 0.001, 0.001, 0.001, -0.001, -0.001, -0.001, -0.001]),
    np.array([1., 3., 1.4, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]),
    np.array([1.5, 1., 1., 0.001, 0.001, 0.001, 0.001, -0.001, -0.001, -0.001]),
    np.array([1.61307398, 1.20354005, 1.24111589, 8.81952541e-04, -6.48578657e-04, 
              3.51944254e-04, 4.14436699e-05, 8.59153391e-04, -1.70811717e-04, -6.19871227e-04]),
    np.array([-1.34084882, 1.78725668, 1.20805872, 4.51291291e-04, 5.75905947e-04, 
              3.45811334e-04, 8.16050092e-04, 2.12146374e-04, -6.37836095e-04, -4.68739581e-04]),
    np.array([1.6, 1., 1.02158901, -2.85475923e-04, -2.36661149e-04, 
              9.50059027e-04, 9.52208643e-04, -8.45302816e-04, 1.31390656e-04, -6.67949121e-04]),
    np.array([-1.47097551, 1.34677568, 1.20403039, -7.30086075e-04, -8.70862525e-04, 
              7.07543827e-04, -1.94519710e-04, -1.80625561e-04, 5.53913815e-04, 9.10504329e-04]),
    np.array([-1.71537874, 2.44335244, 1.46160644, -4.84628964e-04, 7.14558262e-04, 
              9.38953421e-04, -2.79352447e-04, -6.87336201e-05, -5.84337307e-04, 9.77690832e-04]),
    np.array([-1.60851273, 2.16588722, 1.38097482, -4.07575821e-04, 4.00385190e-04, 
              3.07901248e-04, -8.74778461e-04, 2.45939679e-04, 8.91267063e-04, 8.47268430e-04]),
    np.array([-1.36737268, 2.88271015, 1.48649437, 2.02694831e-04, -6.57258878e-04, 
              -1.13491042e-04, 7.77860203e-04, 6.81344067e-04, 6.41901408e-04, 2.28578616e-04]),
    np.array([-1.29382094, 0.95, 1.10975137, -6.54304150e-04, 3.02708499e-04, 
              1.60718466e-04, -8.12299266e-04, -8.79364004e-04, -6.42443978e-04, 9.01851822e-04]),
    np.array([-1.39590546, 2.94564206, 1.49228515, -3.82756775e-04, 5.93724258e-04, 
              -7.61641806e-04, 6.09836073e-04, 5.10959898e-04, -4.97125913e-04, 2.34471366e-04]),
    np.array([-1.36468461, 2.55195862, 1.42169291, -1.27241989e-04, -2.78517746e-04, 
              5.04172499e-04, 7.38056544e-04, 7.46456515e-05, -8.69925008e-04, 9.81464019e-04]),
    np.array([-1.76436012, 2.87261109, 1.22441209, 1.02962855e-04, 1.23681998e-04, 
              -5.61430850e-04, 4.34758068e-04, -5.70585013e-04, -1.15270628e-04, -2.51720155e-05]),
    np.array([-1.50368540, 1., 1.12272043, -2.51225175e-05, 8.25710192e-06, 
              -8.76879956e-05, -8.43472772e-04, 1.55166766e-04, -9.48834834e-04, -2.95726656e-05]),
]

# ours; nonrobust; CE
# avg: 100, 0; 29.47, 92.8; 100, 0

eval_idx = 0
x0 = x0_all[eval_idx]
xG = xG_all[eval_idx] 

obstacles = [(np.array([1., 1.]), 0.2), (np.array([-1.5, 4.]), 0.2), (np.array([0.5, 3.]), 0.2),
                 (np.array([-0.5, 1.]), 0.2), (np.array([-1., 2.5]), 0.2)]
obstacle_centers = np.stack([obs[0] for obs in obstacles], axis=0)
obstacle_r2 = np.array([obs[1] * obs[1] for obs in obstacles])

delay = True
N = 35 
data = np.load(f"data/quad10d_perception_{eval_idx}.npz")
# data = np.load(f"data/quad10d_perception_nonrobust_{eval_idx}.npz")
# data = np.load(f"data/quad10d_perception_CE_{eval_idx}.npz")
primal_x = data["nominal_traj"]
primal_u = data["nominal_input"]
Phi_xx = data["Phi_xx"]
Phi_ux = data["Phi_ux"]
Phi_xy = data["Phi_xy"]
Phi_uy = data["Phi_uy"]
tube = data["backoff"]
tube_f = data["backoff_f"]
tubes = np.vstack([tube[:,:m.nx], tube_f[:m.nx]])
K_mat = Phi_uy - Phi_ux @ np.linalg.inv(Phi_xx) @ Phi_xy

ROLLOUT_NUM = 50
inference_times = []
avg_inference_time = 0.
rollout_states = np.zeros((ROLLOUT_NUM, m.nx, N+1))
rollout_states_open = np.zeros((ROLLOUT_NUM, m.nx, N+1))
feedback_error = 0.
feedback_error_terminal = 0.
openloop_error = 0.
openloop_error_terminal = 0.

success = 0
constr_violation = 0

for rollout in range(ROLLOUT_NUM):
    print("rollout"+str(rollout))
    states = np.zeros((m.nx, N+1))
    states[:,0] = x0
    observations = np.zeros((m.ny, N+1))
    disturbance_state = np.zeros((m.nx, N))
    disturbance_obs = np.zeros((m.ny, N))
    
    if rollout < m.nx:
        disturbance_state[rollout, :] = -1.
    elif rollout < m.nx*2:
        disturbance_state[rollout-m.nx, :] = 1.
    else:
        for i in range(N):
            disturbance_state[:,i] = (np.random.random(m.nx)-0.5) 
            disturbance_state[:,i] = disturbance_state[:,i] / np.linalg.norm(disturbance_state[:,i])

    for t in range(1, N+1):
        delta_u = np.zeros((m.nu))
        if delay:
            # TODO: add a x(-1) to compute y(0) in this case
            start = time.perf_counter()
            if t == 1:
                obs, rgb_im = dino_fn(states[:,t-1])
                observations[:,t-1] = obs[0]
            else:
                obs, rgb_im = dino_fn(states[:,t-2])
                observations[:,t-1] = obs[0]
            end = time.perf_counter()
            inference_times.append(end-start)
                
            delta_u += K_mat[(t-1)*m.nu:t*m.nu, 0:m.ny] @ (observations[:,0] - m.C @ (primal_x[:,0]))# + state_offset))
            for j in range(1, t):
                delta_u += K_mat[(t-1)*m.nu:t*m.nu, j*m.ny:(j+1)*m.ny] @ (observations[:,j] - m.C @ (primal_x[:,j-1]))# + state_offset))
        else: # unused
            observations[:,t-1] = m.measurement(states[:,t-1], disturbance_obs[:,t-1])
            for j in range(t):
                delta_u += K_mat[(t-1)*m.nu:t*m.nu, j*m.ny:(j+1)*m.ny] @ (observations[:,j] - m.C @ primal_x[:,j])
        
        u = primal_u[:,t-1] + delta_u
        
        states[:,t] = np.squeeze(m.ddyn(states[:,t-1], u, disturbance_state[:,t-1], m.dt))
        feedback_error += np.linalg.norm(states[:,t] - primal_x[:,t])
        # plt.imsave(f"video_frames_nonrobust/frame_{t-1:03d}.png", rgb_im)
        
    rollout_states[rollout, :, :] = states.copy()
    feedback_error_terminal += np.linalg.norm(states[:,-1] - primal_x[:,-1])

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

    path_states = states[:, :-1]
    final_state = states[:, -1]
    path_in_bounds = np.all((path_states >= x_min[:, None]-1e-3) & (path_states <= x_max[:, None]+1e-3))
    final_in_bounds = np.all((final_state >= x_min_f-1e-3) & (final_state <= x_max_f+1e-3))
    arrive = np.all((final_state[:3] >= x_min_f[:3]-1e-3) & (final_state[:3] <= x_max_f[:3]+1e-3))

    xy_states = states[:2, :].T
    delta_obs = xy_states[:, None, :] - obstacle_centers[None, :, :]
    collides = np.any(np.sum(delta_obs * delta_obs, axis=2) <= obstacle_r2[None, :]+1e-4)

    success += int(arrive and not collides)
    constr_violation += int((not path_in_bounds) or (not final_in_bounds) or collides)

avg_inference_time = np.mean(inference_times)
np.savez(f"data/rollout_{eval_idx}.npz", 
         rollout_states=rollout_states,
         rollout_states_open=rollout_states_open,
         feedback_error=feedback_error,
         feedback_error_terminal=feedback_error_terminal,
         openloop_error=openloop_error,
         openloop_error_terminal=openloop_error_terminal,
         avg_inference_time=avg_inference_time
         )

env.disconnect()

print(f"test {eval_idx}")
print("success rate: {:.2f}%".format(success/ROLLOUT_NUM * 100))
print("constraint violation rate: {:.2f}%".format(constr_violation/ROLLOUT_NUM * 100))