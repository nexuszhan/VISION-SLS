from dyn.G1.G1_of import G1OF
from solver.nlp_multi_goal import NLP
import numpy as np

data = np.load("data/reference_traj_G1_extracted.npz")
x_ref = data["traj"]

N = x_ref.shape[0] - 1
m = G1OF()

Q = np.diag(
            [0.0, 100.0, 0.0]  # x, y, z
            + [10.0, 100.0, 10.0, 100.0]  # axis-angle residual rotation s.t. R = R_ref * r
            + [75.0] * (m.nq - 7)  # joint positions
            + [0.0, 100.0, 0.0]  # base linear velocities
            + [100.0, 100.0, 100.0]  # base angular velocities
            + [0.5] * (m.ndq - 6)  # joint velocities
        )
R = np.eye(m.nu) * 1e-5
Qf = np.diag(
            [0.0, 100.0, 50.0]  # x, y, z
            + [100.0] * 4  # axis-angle residual rotation s.t. R = R_ref * r
            + [100.0] * (m.nq - 7)  # joint positions
            + [0.0, 100.0, 50.0]  # base linear velocities
            + [100.0, 100.0, 100.0]  # base angular velocities
            + [0.0] * (m.ndq - 6)  # joint velocities
            )

solver = NLP(N, Q, R, m, Qf, [])

x0 = x_ref[0,:]

goals = []
for i in range(1, x_ref.shape[0]):
    goals.append(x_ref[i,:])

sol = solver.solve(x0, goals)

print("Unitree G1: ", sol["success"])

np.savez("data/G1_4.npz", 
         initial_traj=np.array(sol["primal_x"]),
         initial_input=np.array(sol["primal_u"]),
         initial_vec=np.array(sol["primal_vec"]),
         initial_dual_vec=np.array(sol["dual_vec"]),
         initial_cost=sol["cost"]
    )
