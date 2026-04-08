from pydrake.solvers import IpoptSolver, SnoptSolver, GurobiSolver, MosekSolver, \
    MathematicalProgram, SolverOptions, CommonSolverOption
from pydrake.math import eq, le, ge, matmul
from pydrake.all import Variables
import pydrake.symbolic as sym
import pydrake
import numpy as np
import scipy.io as sio
from scipy.linalg import block_diag
import os, sys
import matplotlib.pyplot as plt
from matplotlib import patches
import time
from omegaconf import DictConfig, OmegaConf
from perception_utils.losses import *
from perception_utils.networks import *
from perception_utils.datasets import *
from perception_utils.utils import *
from perception_utils.train import train_network
import cvxpy as cp

base_path = os.path.dirname(__file__)

if sys.platform == 'linux':
    os.environ["GRB_LICENSE_FILE"] = "/home/nexus/gurobi.lic"
    # os.environ["MOSEKLM_LICENSE_FILE"] = "/home/gchou/mosek/mosek.lic"
else:
    os.environ["GRB_LICENSE_FILE"] = "/Users/glenchou/gurobi.lic"
    os.environ["MOSEKLM_LICENSE_FILE"] = "/Users/glenchou/mosek/mosek.lic"

def visualize_fit(load_path):
    load_path_orig = load_path
    load_path = os.path.join(base_path, load_path)
    loaded = np.load(os.path.join(base_path, load_path),
                     allow_pickle=True)
    loaded_data = np.load(os.path.join(base_path, "../../data/checkpoints/unicycle/unicycle_observability_bound_ppoly_data.npz"),
                     allow_pickle=True)
    theta_val, theta_bias_val, degree = loaded['theta_poly'][:, None], loaded['theta_bias'][0], loaded['degree']
    data_x, mons_val = loaded_data['data_x'], loaded_data['mons_val']

    # prog = MathematicalProgram()
    # x = sym.MakeVectorContinuousVariable(2, "x")
    # V = prog.NewFreePolynomial(Variables(x), degree)
    # mons = list(V.monomial_to_coefficient_map().keys())
    # mons_val = []
    # for i in range(len(mons)):
    #     mons_val.append(mons[i].Evaluate(x, data_x.T))
    # mons_val = np.array(mons_val)

    fit_vals = (theta_val[:, None].T @ mons_val).clip(min=theta_bias_val)
    # import matplotlib as mpl
    # mpl.use('Qt5Agg')  # or can use 'TkAgg', whatever you have/prefer


    plt.figure(), plt.scatter(data_x[:, 0], data_x[:, 1], c=fit_vals), plt.colorbar()
    plt.title('theta_bias = ' + str(theta_bias_val) + ', degree = ' + str(degree))
    plt.xlabel(load_path_orig)
    plt.savefig(load_path + '.png')
    sio.savemat(load_path + '.mat', {'theta_val': theta_val, 'theta_bias_val': theta_bias_val, 'degree': degree,
                                     'fit_vals': fit_vals, 'data_x': data_x, 'mons_val': mons_val})
    # plt.show()
    
def main():
    params = OmegaConf.load("config/learn_model_error_observability.yaml")
    save_path = os.path.join(base_path, "checkpoints/turtlebot_error_bound")
    # load data
    if torch.has_cuda and params.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # load dataset
    data = CustomDataset(os.path.join(base_path, params.dataset.load_filename),
                         None,#os.path.join(base_path, params.dataset.load_dino_filename),
                         device, params.model.z_dim)
    # data.data_x = torch.hstack([data.data_x, torch.randn(data.data_x.shape[0], 1)]).float().to(device)
    data.data_x = data.data_x.to(device).float()
    data.data_z = data.data_z.to(device).float()
    # Load prediction model
    model = SupervisedDinoObservability(params.model).to(device)
    checkpoint = torch.load(os.path.join(base_path, params.train.load_ckpt), map_location=device)
    model.load_state_dict(checkpoint)
    # Evaluate errors
    C_norm = model.C / model.C.norm(dim=1, keepdim=True)
    print(C_norm)

    error = torch.abs(model(data.data_z) - (C_norm @ data.data_x.T).T)
    data_error = dcn(error)
    print(np.sum(data_error[:,0] > 0.05))
    print(np.sum(data_error[:,1] > 0.05))
    print(np.sum(data_error[:,2] > 0.05))
    # print(data_error)

    # error = torch.norm(model(data.data_z) - (C_norm @ data.data_x.T).T, dim=1).to(device)[:, None]
    error = torch.abs(model(data.data_z) - (C_norm @ data.data_x.T).T).max(dim=1)[0].to(device)[:, None]
    data_err = CustomDatasetDinoErr(os.path.join(base_path, params.dataset.load_filename), error, device, params.model.z_dim)
    data_err.data_x = data_err.data_x.to(device).float()[:, :params.model.x_dim_error]
    data_err.data_err = data_err.data_err.detach().to(device).float().reshape(data_err.data_err.shape[0], 1)
    data_error = dcn(error)
    data_x = dcn(data_err.data_x)
    inds_keep = np.random.choice(np.arange(data_x.shape[0]), 500, replace=False)
    data_error = data_error[inds_keep]
    data_x = data_x[inds_keep]
    # print(data_error)
    data_error = data_error.reshape(-1)
    # print(data_error.shape)
    inds_not_keep = np.setdiff1d(np.arange(data_x.shape[0]), inds_keep)
    data_x_eval = data_x[inds_not_keep]
    data_error_eval = data_error[inds_not_keep]

    # form optimization problem
    # define variables
    # degree = 2
    # THRESH_KEEP = 0.975
    # THETA_BIAS_MAX = 0.04
    # degrees = [2, 4, 6, 8]
    # THRESH_KEEPS = [0.95, 0.975, 0.99, 0.995]
    # THETA_BIAS_MAXS = [0.005, 0.01, 0.02, 0.04]
    # degrees = [4, 6]
    # THRESH_KEEPS = [0.9, 0.95, 0.97, 0.99]
    # THETA_BIAS_MAXS = [0.02, 0.05]
    degrees = [3, 4]
    THRESH_KEEPS = [0.9, 0.95, 0.97]
    THETA_BIAS_MAXS = [0.02, 0.04, 0.06, 0.08]
    nx = 3
    for degree in degrees:
        for THRESH_KEEP in THRESH_KEEPS:
            for THETA_BIAS_MAX in THETA_BIAS_MAXS:
                print("--"*10)
                print("degree: {:}, thres_keep: {:.2f}, thres_bias_max: {:.2f} ".format(degree, THRESH_KEEP, THETA_BIAS_MAX))

                prog = MathematicalProgram()
                M = 1.001*data_error.max()
                relax_poly = True
                x = sym.MakeVectorContinuousVariable(nx, "x")
                V = prog.NewFreePolynomial(Variables(x), degree)
                mons = list(V.monomial_to_coefficient_map().keys())
                # theta = prog.NewContinuousVariables(len(mons), "theta")
                # theta_bias = prog.NewContinuousVariables(1, "theta_bias")
                # z_var = prog.NewBinaryVariables(len(data_error), "z_var")
                # q_var = prog.NewBinaryVariables(len(data_error), "q_var")

                mons_val = []
                for i in range(len(mons)):
                    mons_val.append(mons[i].Evaluate(x, data_x.T))
                mons_val = np.array(mons_val)
                lamb = len(data_error) * 5.

                N = len(data_error)
                n_mons = mons_val.shape[0]

                # Decision variables
                theta = cp.Variable(n_mons) 
                theta_bias = cp.Variable() 
                z_var = cp.Variable(N, boolean=True)
                q_var = cp.Variable(N, boolean=True)

                constraints = []

                # theta[:, None].T @ mons_val  -->  (1, n_mons) @ (n_mons, N) = (1, N)
                # In CVXPY: theta @ mons_val is shape (N,) (since theta is (n_mons,))
                pred = theta @ mons_val              # shape (N,)

                if relax_poly:
                    # theta[:, None].T @ mons_val >= data_error.T - M*z_var - M*q_var
                    constraints.append(pred >= data_error - M * z_var - M * q_var)
                else:
                    constraints.append(pred >= data_error - M * z_var)

                # theta_bias * ones((1, N)) >= data_error.T - M*(1 - z_var) - M*q_var
                constraints.append(theta_bias * np.ones(N) >= data_error - M * (1.0 - z_var) - M * q_var)

                # Relationship between q_var and z_var
                if relax_poly:
                    constraints.append(cp.sum(q_var) == (1.0 - THRESH_KEEP) * cp.sum(z_var))
                else:
                    constraints.append(cp.sum(q_var) <= (1.0 - THRESH_KEEP) * cp.sum(z_var))

                # Bounds on theta_bias
                constraints.append(theta_bias >= 0)
                constraints.append(theta_bias <= THETA_BIAS_MAX)

                # Bounds on theta
                constraints.append(theta >= -0.5)
                constraints.append(theta <= 0.5)

                # Objective:
                #   objective = matmul(theta[:,None].T, mons_val).sum() + lamb * theta_bias[0]
                #   In CVXPY: cp.sum(theta @ mons_val) + lamb * theta_bias
                objective = cp.Minimize(cp.sum(theta @ mons_val) + lamb * theta_bias)

                prob = cp.Problem(objective, constraints)

                # Solve with Gurobi; TimeLimit matches your pydrake options
                prob.solve(solver=cp.GUROBI, verbose=False, TimeLimit=120)
                
                # if prob.status not in ["optimal", "optimal_inaccurate"]:
                #     continue
                if prob.status in ["infeasible", "unbounded"]:
                    continue
                
                obj_val = prob.value
                theta_val = np.array(theta.value)
                theta_bias_val = theta_bias.value
                z_val = z_var.value
                q_val = q_var.value

                pred_val = theta_val @ mons_val
                print("objective {:.3f} with {:} preds smaller than actual error".format(obj_val, np.sum(pred_val < data_error)))

                mons_eval = []
                for i in range(len(mons)):
                    mons_eval.append(mons[i].Evaluate(x, data_x_eval.T))
                mons_eval = np.array(mons_eval)
                print("eval: {:} preds smaller than actual error".format(np.sum(theta_val @ mons_eval < data_error_eval)))

    #             objective = matmul(theta[:,None].T, mons_val).sum() + lamb * theta_bias[0]
    #             if relax_poly:
    #                 prog.AddConstraint(ge(theta[:, None].T @ mons_val, data_error.T - M * z_var - M * q_var))
    #             else:
    #                 prog.AddConstraint(ge(theta[:, None].T @ mons_val, data_error.T - M * z_var))
    #             prog.AddConstraint(ge(theta_bias * np.ones((1, len(data_error))), data_error.T - M * (1. - z_var) - M * q_var))
    #             # prog.AddConstraint(le(q_var, z_var))
    #             if relax_poly:
    #                 prog.AddConstraint(q_var.sum() == (1. - THRESH_KEEP) * z_var.sum())
    #             else:
    #                 prog.AddConstraint(q_var.sum() <= (1. - THRESH_KEEP) * z_var.sum())
    #             prog.AddConstraint(theta_bias[0] >= 0)
    #             prog.AddConstraint(theta_bias[0] <= THETA_BIAS_MAX)
    #             prog.AddConstraint(ge(theta, -0.5))
    #             prog.AddConstraint(le(theta, 0.5))
    #             prog.AddCost(objective)
    #             solver = GurobiSolver()
    #             options = SolverOptions()
    #             options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    #             options.SetOption(GurobiSolver.id(), 'TimeLimit', 120)
    #             prog.SetSolverOptions(options)
    #             result = solver.Solve(prog)
    #             theta_val = result.GetSolution(theta)
    #             theta_bias_val = result.GetSolution(theta_bias)
    #             z_val = result.GetSolution(z_var)
    #             q_val = result.GetSolution(q_var)

                theta_val, theta_bias_val, degree, nx, z_val, q_val, M = \
                    theta_val.astype('float64'), theta_bias_val.astype('float64'), degree, nx, z_val.astype('float64'), \
                        q_val.astype('float64'), M
                save_path_curr = save_path + str(degree) + '_' + str(THRESH_KEEP) + '_' + str(THETA_BIAS_MAX) + '.npz'
                np.savez(save_path_curr, theta_poly=theta_val, theta_bias=theta_bias_val, degree=degree, nx=nx, z_val=z_val, q_val=q_val,
                         M=M)
    #             fit_vals = (theta_val[:, None].T @ mons_val).clip(min=theta_bias_val)
    #             # import matplotlib as mpl
    #             # mpl.use('Qt5Agg')  # or can use 'TkAgg', whatever you have/prefer

    #             plt.figure(), plt.scatter(data_x[:, 0], data_x[:, 1], c=fit_vals), plt.colorbar()
    #             plt.title('theta_bias = ' + str(theta_bias_val) + ', degree = ' + str(degree))
    #             plt.savefig(save_path + str(degree) + '_' + str(THRESH_KEEP) + '_' + str(THETA_BIAS_MAX) + '.png')
    #             sio.savemat(save_path + str(degree) + '_' + str(THRESH_KEEP) + '_' + str(THETA_BIAS_MAX) + '.mat',
    #                         {'theta_val': theta_val, 'theta_bias_val': theta_bias_val, 'degree': degree,
    #                          'fit_vals': fit_vals, 'data_x': data_x, 'mons_val': mons_val})
    # print('done')

if __name__ == '__main__':
    main()
