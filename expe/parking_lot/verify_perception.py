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
dino_feature_reduction = SupervisedDinoObservabilityLargeUnicycle(params.model).to(device)
if params.train.load_ckpt_bound is not None:
    checkpoint = torch.load(os.path.join(base_path, "data/unicycle_checkpoint_observability_ft_075w_3pi4_097_300000_post.pt"), map_location=device)
    dino_feature_reduction.load_state_dict(checkpoint)

loaded = np.load(os.path.join(base_path, "checkpoints/error_bound_3_0.97_0.04_250.npz"),
                 allow_pickle=True)
theta_poly, theta_bias, degree = loaded['theta_poly'][:,None], loaded['theta_bias'][0], loaded['degree']

prog = MathematicalProgram()
x_red = sym.MakeVectorContinuousVariable(loaded['nx'], "x_red")
V = prog.NewFreePolynomial(Variables(x_red), degree)
mons = list(V.monomial_to_coefficient_map().keys())
mons_expr = np.array([mons[i].ToExpression() for i in range(len(mons))])
mons_fn, _ = sym_to_pytorch(mons_expr, x_red)

jac_mons_expr = np.array([mons_expr[i].Jacobian(x_red) for i in range(len(mons_expr))])
jac_mons_fn, _ = sym_to_pytorch(jac_mons_expr, x_red)

def cov_fn(x, scale=1.):
    if torch.is_tensor(x):
        mons_val = mons_fn(x).T
        return scale*(torch.tensor(theta_poly.T).float() @ mons_val).clip(min=torch.tensor(theta_bias).float()).T
    else:
        mons_val = dcn(mons_fn(torch.tensor(x).float()).T)
        return scale*(theta_poly.T @ mons_val).clip(min=theta_bias).T

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
robot_env = racecar.Racecar(env, urdfRootPath=urdfRoot, timeStep=0.001) # gym envrionment?
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

size_validation_set = 500 #1000
from dyn.unicycle_of_perception import Unicycle_OF
m = Unicycle_OF()
nx = m.nx
ny = m.ny

# storage
X_eval      = np.zeros((nx, size_validation_set), dtype=float)   # evaluated states (with offset)
Y_hat       = np.zeros((ny, size_validation_set), dtype=float)   # model (NN) observations
Y_lin       = np.zeros((ny, size_validation_set), dtype=float)   # linear Cx observations
Res         = np.zeros((ny, size_validation_set), dtype=float)   # residuals (y_hat - y_lin)
Norm_res    = np.zeros((size_validation_set,), dtype=float)      # ||residual||
V_vals      = np.zeros((size_validation_set,), dtype=float)      # V(residual)

over_bound = 0
for t in range(size_validation_set):
    x = np.random.uniform(np.array([0., -1.5, -np.pi*3/4, 0.]), np.array([3., 1.5, np.pi*3/4, 0.]))

    x_eval = x

    y_hat, _ = dino_fn(x_eval)
    y_hat = np.asarray(y_hat[0], dtype=float)

    # print("reduced model observation: ", y_hat)
    y_lin = (m.C @ x_eval).astype(float)
    # print("linear model observation: ", y_lin)

    residual = y_hat - y_lin        

    V_val = float(cov_fn(x[:2]))
    # assert abs(V_val - m.get_disturbance(x-np.array([3.,0.,0.,0.]))) < 1e-3
    # store
    X_eval[:, t] = x_eval
    Y_hat[:, t]  = y_hat
    Y_lin[:, t]  = y_lin
    Res[:, t]    = residual
    Norm_res[t]  = np.max(residual) #np.linalg.norm(residual)
    V_vals[t]    = V_val

    # print(f"[t={t}] ||res||={np.linalg.norm(residual):.6g}, V(residual)={V_val:.6g}")
    # print(f"[t={t}] ||res||={np.max(residual):.6g}, V(residual)={V_val:.6g}")

    if Norm_res[t] > V_vals[t]:
        print(f"evaluated state: ", x)
        print("reduced model observation: ", y_hat)
        print("linear model observation: ", y_lin)
        print(f"[t={t}] ||res||={np.max(residual):.6g}, V(residual)={V_val:.6g}")
        print("wrong error bound")
    else:
        over_bound += 1

env.disconnect()

print("accuracy: {:.2f}%".format(over_bound/size_validation_set*100))

# save everything
os.makedirs("data", exist_ok=True)
out_path = os.path.join(base_path, "checkpoints", "error_bound_250_val.npz")
np.savez(out_path,
         X_eval=X_eval, Y_hat=Y_hat, Y_lin=Y_lin,
         Res=Res, Norm_res=Norm_res, V_vals=V_vals,
         ny=ny, nx=nx)
print(f"Saved to {out_path}")

# 50 91.4%
# 100 92.2%
# 250 97.4%
# 500 99%
# 1500 99%
# 2500 99.2%