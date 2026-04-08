"""
It's recommended to install pinocchio in another environment and run this script.
After getting the G1_ddyn.c, run the following command to get .so file
gcc -fPIC -shared G1_ddyn.c -o G1_ddyn.so
"""

import numpy as np
from model import Model
import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin 
from pinocchio.visualize import MeshcatVisualizer as Mviz
import time, os

def make_rk4_dynamics(model, data, dt):
    nq, nv = model.nq, model.nv
    nu = nv - 6

    # selection: joint torques -> generalized torques
    S = ca.SX.zeros(nv, nu)
    S[6:, :] = ca.SX.eye(nu)

    # symbols
    q  = ca.SX.sym("q",  nq)
    v  = ca.SX.sym("v",  nv)
    u  = ca.SX.sym("u",  nu)

    # forward dynamics a(q,v,u) using ABA (symbolic)
    # gravity etc. should be set in `model` before creating this function
    # tau_full = S @ u
    # a = cpin.aba(model, cpin.Data(model), cpin.normalize(model, q), v, tau_full)

    def acc(q_, v_, u_):
        # one-line helper returning a(q,v,u) (SX)
        return cpin.aba(model, data, cpin.normalize(model, q_), v_, S @ u_)

    # ---- RK4 on [q; v] with manifold integration for q ----
    # k1
    k1_q = v
    k1_v = acc(q, v, u)

    # k2 at x + 0.5*dt*k1
    q2 = cpin.integrate(model, q, (0.5*dt) * k1_q)
    v2 = v + 0.5*dt * k1_v
    k2_q = v2
    k2_v = acc(q2, v2, u)

    # k3 at x + 0.5*dt*k2
    q3 = cpin.integrate(model, q, (0.5*dt) * k2_q)
    v3 = v + 0.5*dt * k2_v
    k3_q = v3
    k3_v = acc(q3, v3, u)

    # k4 at x + dt*k3
    q4 = cpin.integrate(model, q, dt * k3_q)
    v4 = v + dt * k3_v
    k4_q = v4
    k4_v = acc(q4, v4, u)

    # combine
    dq_rk = (dt/6.0) * (k1_q + 2*k2_q + 2*k3_q + k4_q)
    dv_rk = (dt/6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

    q_next = cpin.integrate(model, q, dq_rk)
    q_next = cpin.normalize(model, q_next)   # keep quaternion well-formed
    v_next = v + dv_rk

    x = ca.vertcat(q, v)
    x_next = ca.vertcat(q_next, v_next)

    return ca.Function("G1_ddyn", [x, u], [x_next], ["x", "u"], ["x_next"])

if __name__ == "__main__":
    urdf = "g1_23dof.urdf"
    base_path = os.path.dirname(__file__)

    model = pin.buildModelFromUrdf(urdf, pin.JointModelFreeFlyer())
    geom_model = pin.buildGeomFromUrdf(model, urdf, pin.COLLISION, package_dirs=base_path)
    visual_model = pin.buildGeomFromUrdf(model, urdf, pin.VISUAL, package_dirs=base_path)
    data  = model.createData()

    for j in range(1, model.njoints):  # joint 0 is universe
        Ji = model.joints[j]
        name = model.names[j]
        iq = model.idx_qs[j]   # starting index in q
        iv = model.idx_vs[j]   # starting index in v
        print(f"{j:2d} {name:>28s}  nq_j={model.nqs[j]} nv_j={model.nvs[j]}  q[{iq}:{iq+model.nqs[j]}]  v[{iv}:{iv+model.nvs[j]}]")

    joint_pos_lb = model.lowerPositionLimit[7:]
    joint_pos_ub = model.upperPositionLimit[7:]

    cmodel = cpin.Model(model)  
    cdata = cmodel.createData()
    
    ddyn_fun = make_rk4_dynamics(cmodel, cdata, 0.01).expand()
    fwd1 = ddyn_fun.forward(1)
    rev1 = ddyn_fun.reverse(1)
    fwd2 = ddyn_fun.forward(2)
    rev2 = ddyn_fun.reverse(2)

    f_r1_f1 = rev1.forward(1)   # needed for "adj1_*" forward calls
    f_f1_r1 = fwd1.reverse(1)

    # (optional but harmless) more:
    f_r1_r1 = rev1.reverse(1)
    f_f1_f1 = fwd1.forward(1)
    cg = ca.CodeGenerator("G1_ddyn.c")
    cg.add(ddyn_fun)
    cg.add(fwd1)
    cg.add(rev1)
    cg.add(fwd2)
    cg.add(rev2)
    cg.add(f_r1_f1)
    cg.add(f_f1_r1)
    cg.add(f_r1_r1)
    cg.add(f_f1_f1)
    cg.generate()
