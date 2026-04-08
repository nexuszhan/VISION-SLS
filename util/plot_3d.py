# util/plot_3d.py
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

def _cylinder_mesh(cx, cy, r, zmin, zmax, n=40):
    theta = np.linspace(0, 2*np.pi, n)
    z = np.linspace(zmin, zmax, n)
    theta, z = np.meshgrid(theta, z)
    X = cx + r*np.cos(theta)
    Y = cy + r*np.sin(theta)
    Z = z
    return X, Y, Z

def _box_faces(center, half):
    """Return the 6 faces (quads) of an axis-aligned box."""
    cx, cy, cz = center
    hx, hy, hz = np.abs(half)
    v0 = (cx - hx, cy - hy, cz - hz)
    v1 = (cx + hx, cy - hy, cz - hz)
    v2 = (cx + hx, cy + hy, cz - hz)
    v3 = (cx - hx, cy + hy, cz - hz)
    v4 = (cx - hx, cy - hy, cz + hz)
    v5 = (cx + hx, cy - hy, cz + hz)
    v6 = (cx + hx, cy + hy, cz + hz)
    v7 = (cx - hx, cy + hy, cz + hz)
    return [
        [v0, v1, v2, v3],  # bottom
        [v4, v5, v6, v7],  # top
        [v0, v1, v5, v4],  # -y
        [v2, v3, v7, v6],  # +y
        [v1, v2, v6, v5],  # +x
        [v0, v3, v7, v4],  # -x
    ]

def plot_3d_obstacles(obstacles, z_min, z_max, ax, cyl_res=40):
    # obstacles as vertical cylinders (optional)
    for i in (2, 4):
        c, r = obstacles[i]
        Xc, Yc, Zc = _cylinder_mesh(c[0], c[1], r, z_min+0.1, z_max, n=cyl_res)
        ax.plot_surface(Xc, Yc, Zc, color='red', alpha=0.25, linewidth=0, shade=True, zorder=0)
    
    # ---- x–y footprints on the same 3D axes (projected to z = const) ----
    # place slightly below the data range to avoid z-fighting
    z_plane = z_min - 0.02 * (z_max - z_min)
    z0, z1 = min(z_plane, z_min), z_max
    ax.set_zlim(z0, z1)

    # obstacles as circles on the floor
    th = np.linspace(0, 2 * np.pi, 200)
    circle_color = 'r'
    for i in (2,4):
        c, r = obstacles[i]
        cx, cy = c
        ax.plot(cx + r * np.cos(th), cx * 0 + cy + r * np.sin(th),
                zs=z_plane, zdir='z', lw=1.6, alpha=0.9, color=circle_color)
        
    from matplotlib.lines import Line2D
    obstacle_proxy = Line2D([0], [0], color=circle_color, lw=1.6, label='Obstacle')

    return obstacle_proxy

def plot_3d_rollout(X_fast, hf, fast_c, label_prefix,
                    x_min, x_max, y_min, y_max, z_min, z_max, 
                    ax, rollouts=None, subsample=2, ellipsoid_res=18, cyl_res=40):
    """
    X_fast, X_init: (nx, T)
    backoff_fast, backoff_init: (T, nx) or (T, >=3); uses indices 0:3 as half-sizes
    obstacles: list of ((cx,cy), radius)
    """
    px, py, pz = X_fast[0, :], X_fast[1, :], X_fast[2, :]

    # fig = plt.figure(figsize=(8, 6),layout='constrained')
    # ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=38, azim=-120, roll=3)   # mpl ≥ 3.7

    from matplotlib.patches import Patch
    from matplotlib.colors import to_rgba

    # consistent colors
    box_fast_fc = to_rgba(fast_c, 0.18)

    T = X_fast.shape[1]

    # trajectories
    (line_fast,) = ax.plot(px, py, pz, lw=2, color=fast_c, label=label_prefix+" trajectory")
    line_rollout = None
    if rollouts is not None:
        for i in range(rollouts.shape[0]):
            (line_rollout,) = ax.plot(rollouts[i, 0, :T+1], rollouts[i, 1, :T+1], rollouts[i, 2, :T+1], lw=2, color=fast_c, linestyle='--',  label=label_prefix+" rollouts")

    # --- reachable sets as HYPERBOXES (no subsampling) ---
    # fast (green-ish)
    for i in range(T):
        half = hf[i]
        if np.all(half == 0):
            continue
        faces = _box_faces(X_fast[:3, i], half)
        ax.add_collection3d(Poly3DCollection(
            faces, facecolors=box_fast_fc, edgecolors='none', linewidths=0))
    # -----------------------------------------------------

    # ---- x–y footprints on the same 3D axes (projected to z = const) ----
    # place slightly below the data range to avoid z-fighting
    z_plane = z_min - 0.02 * (z_max - z_min)
    z0, z1 = min(z_plane, z_min), z_max
    ax.set_zlim(z0, z1)

    # trajectory footprints
    ax.plot(px, py, zs=z_plane, zdir='z', lw=1.4, alpha=0.9, color=fast_c)
    if rollouts is not None:
        for i in range(rollouts.shape[0]):
            ax.plot(rollouts[i, 0, :T+1], rollouts[i, 1, :T+1], zs=z_plane, zdir='z', lw=1.4, alpha=0.9, color=fast_c, linestyle='--')

    # styling
    fast_face = (0.35, 0.75, 0.35, 0.35)
    init_face = (0.55, 0.40, 0.80, 0.30)
    fast_edge = 'tab:green'
    init_edge = 'tab:purple'
    edge_lw = 1.6
    outline_lw = 2.2
    eps = 1e-3  # lift outlines to avoid z-fighting

    step = max(1, subsample)
    for i in range(0, T, step):
        # nominal box
        hx, hy = hf[i, 0], hf[i, 1]
        if hx > 0 and hy > 0:
            xs = np.array([px[i] - hx, px[i] + hx, px[i] + hx, px[i] - hx])
            ys = np.array([py[i] - hy, py[i] - hy, py[i] + hy, py[i] + hy])
            verts = [list(zip(xs, ys, [z_plane] * 4))]
            ax.add_collection3d(Poly3DCollection(
                verts, facecolors=(0.4, 0.8, 0.4, 0.5), edgecolors='none', zorder=4))
    # ---------------------------------------------------------------------

    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_zlim(z_min, z_max)
    ax.set_xlabel("Position $p_x$ [m]",fontsize=14,labelpad=12)
    ax.set_ylabel("Position $p_y$ [m]", fontsize = 14,labelpad=12)
    ax.set_zlabel("Position $p_z$ [m.]", fontsize = 14,labelpad=18)
    if label_prefix == "Optimal":
        tube_fast = Patch(facecolor=to_rgba(fast_c, 0.25), edgecolor='none', label=label_prefix+" tubes (cost=3120)")
    elif label_prefix == "CE":
        tube_fast = Patch(facecolor=to_rgba(fast_c, 0.25), edgecolor='none', label=label_prefix+" tubes (cost=3130)")
    else:
        tube_fast = Patch(facecolor=to_rgba(fast_c, 0.25), edgecolor='none', label=label_prefix+" tubes")
    
    return line_fast, tube_fast, line_rollout

    #plt.tight_layout()
    #plt.show()  # keep interactive window
    # fig.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.close(fig)


# def plot_3d_rollout(name, X_fast, X_init, backoff_fast, backoff_init, obstacles,
#                     save_path, ax, subsample=2, ellipsoid_res=18, cyl_res=40):
#     """
#     X_fast, X_init: (nx, T)
#     backoff_fast, backoff_init: (T, nx) or (T, >=3); uses indices 0:3 as half-sizes
#     obstacles: list of ((cx,cy), radius)
#     """
#     # positions
#     px, py, pz = X_fast[0, :], X_fast[1, :], X_fast[2, :]
#     px0, py0, pz0 = X_init[0, :], X_init[1, :], X_init[2, :]

#     # per-step half-sizes for boxes (x,y,z)
#     hf = np.abs(backoff_fast[:, :3])
#     hi = np.abs(backoff_init[:, :3])

#     # bounds including boxes
#     x_min = min((px - hf[:, 0]).min(), (px0 - hi[:, 0]).min())
#     x_max = max((px + hf[:, 0]).max(), (px0 + hi[:, 0]).max())
#     y_min = min((py - hf[:, 1]).min(), (py0 - hi[:, 1]).min())
#     y_max = max((py + hf[:, 1]).max(), (py0 + hi[:, 1]).max())
#     z_min = min((pz - hf[:, 2]).min(), (pz0 - hi[:, 2]).min())
#     z_max = max((pz + hf[:, 2]).max(), (pz0 + hi[:, 2]).max())
   
#     # fig = plt.figure(figsize=(8, 6),layout='constrained')
#     # ax = fig.add_subplot(111, projection='3d')
#     ax.view_init(elev=38, azim=-120, roll=3)   # mpl ≥ 3.7

#     from matplotlib.patches import Patch
#     from matplotlib.colors import to_rgba

#     # consistent colors
#     fast_c = "tab:green"
#     init_c = "tab:blue"
#     box_fast_fc = to_rgba(fast_c, 0.18)
#     box_init_fc = to_rgba(init_c, 0.15)

#     # obstacles as vertical cylinders (optional)
#     for i in (2, 4):
#         c, r = obstacles[i]
#         Xc, Yc, Zc = _cylinder_mesh(c[0], c[1], r, z_min+0.1, z_max, n=cyl_res)
#         ax.plot_surface(Xc, Yc, Zc, alpha=0.25, linewidth=0, shade=True, zorder=0)

#     # trajectories
#     (line_fast,) = ax.plot(px, py, pz, lw=2, color=fast_c, label="Optimal trajectory")
#     (line_init,) = ax.plot(px0, py0, pz0, lw=1.5, color=init_c, label="CE trajectory")

#     T = X_fast.shape[1]

#     # --- reachable sets as HYPERBOXES (no subsampling) ---
#     # fast (green-ish)
#     for i in range(T):
#         half = hf[i]
#         if np.all(half == 0):
#             continue
#         faces = _box_faces(X_fast[:3, i], half)
#         ax.add_collection3d(Poly3DCollection(
#             faces, facecolors=box_fast_fc, edgecolors='none', linewidths=0))

#     # init (purple-ish, lighter)
#     for i in range(T):
#         half0 = hi[i]
#         if np.all(half0 == 0):
#             continue
#         faces0 = _box_faces(X_init[:3, i], half0)
#         ax.add_collection3d(Poly3DCollection(
#             faces0, facecolors=box_init_fc, edgecolors='none', linewidths=0))
#     # -----------------------------------------------------

#     # ---- x–y footprints on the same 3D axes (projected to z = const) ----
#     # place slightly below the data range to avoid z-fighting
#     z_plane = z_min - 0.02 * (z_max - z_min)
#     z0, z1 = min(z_plane, z_min), z_max
#     ax.set_zlim(z0, z1)

#     # obstacles as circles on the floor
#     th = np.linspace(0, 2 * np.pi, 200)
#     circle_color = 'k'
#     for i in (2,4):
#         c, r = obstacles[i]
#         cx, cy = c
#         ax.plot(cx + r * np.cos(th), cx * 0 + cy + r * np.sin(th),
#                 zs=z_plane, zdir='z', lw=1.6, alpha=0.9, color='k')

#     from matplotlib.lines import Line2D
#     obstacle_proxy = Line2D([0], [0], color=circle_color, lw=1.6, label='Obstacle')
#     # trajectory footprints
#     ax.plot(px, py, zs=z_plane, zdir='z', lw=1.4, alpha=0.9, color=fast_c)
#     ax.plot(px0, py0, zs=z_plane, zdir='z', lw=1.2, alpha=0.9, color=init_c)

#     # styling
#     fast_face = (0.35, 0.75, 0.35, 0.35)
#     init_face = (0.55, 0.40, 0.80, 0.30)
#     fast_edge = 'tab:green'
#     init_edge = 'tab:purple'
#     edge_lw = 1.6
#     outline_lw = 2.2
#     eps = 1e-3  # lift outlines to avoid z-fighting

#     step = max(1, subsample)
#     for i in range(0, T, step):
#         # nominal box
#         hx, hy = hf[i, 0], hf[i, 1]
#         if hx > 0 and hy > 0:
#             xs = np.array([px[i] - hx, px[i] + hx, px[i] + hx, px[i] - hx])
#             ys = np.array([py[i] - hy, py[i] - hy, py[i] + hy, py[i] + hy])
#             verts = [list(zip(xs, ys, [z_plane] * 4))]
#             ax.add_collection3d(Poly3DCollection(
#                 verts, facecolors=(0.4, 0.8, 0.4, 0.5), edgecolors='none', zorder=4))

#         # initial box
#         hx0, hy0 = hi[i, 0], hi[i, 1]
#         if hx0 > 0 and hy0 > 0:
#             xs0 = np.array([px0[i] - hx0, px0[i] + hx0, px0[i] + hx0, px0[i] - hx0])
#             ys0 = np.array([py0[i] - hy0, py0[i] - hy0, py0[i] + hy0, py0[i] + hy0])
#             verts0 = [list(zip(xs0, ys0, [z_plane] * 4))]
#             ax.add_collection3d(Poly3DCollection(
#                 verts0, facecolors=(0.12, 0.47, 0.71, 0.45),   # blue with alpha
#                 edgecolors='none', zorder=3))
#     # ---------------------------------------------------------------------

#     ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_zlim(z_min, z_max)
#     ax.set_xlabel("Position $p_x$ [m]",fontsize=14,labelpad=12)
#     ax.set_ylabel("Position $p_y$ [m]", fontsize = 14,labelpad=12)
#     ax.set_zlabel("Position $p_z$ [m.]", fontsize = 14,labelpad=18)
#     tube_fast = Patch(facecolor=to_rgba(fast_c, 0.25), edgecolor='none', label="Optimal tubes")
#     tube_init = Patch(facecolor=to_rgba(init_c, 0.25), edgecolor='none', label="CE: post-hoc tubes")
#     ax.legend(handles=[line_fast, line_init, tube_fast, tube_init, obstacle_proxy], loc="upper left",fontsize=12)
#     # tick labels
#     ax.tick_params(axis='both', which='major', labelsize=12)
#     ax.tick_params(axis='z', which='major', labelsize=12)

#     ax.set_box_aspect([1, 1, 0.7])
#     ax.grid(True)

#     # enforce equal x–y scale so circles look like circles
#     xr = x_max - x_min
#     yr = y_max - y_min
#     r = 0.5 * max(xr, yr)
#     xmid = 0.5 * (x_max + x_min)
#     ymid = 0.5 * (y_max + y_min)

#     ax.set_xlim(xmid - r, xmid + r)
#     ax.set_ylim(ymid - r, ymid + r)
#     ax.set_box_aspect([1, 1, 0.7])  # keep x and y visually square

#     #plt.tight_layout()
#     #plt.show()  # keep interactive window
#     # fig.savefig(save_path, dpi=300, bbox_inches='tight')
#     # plt.close(fig)
