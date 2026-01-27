import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

from matplotlib import patches
from matplotlib.animation import FuncAnimation, PillowWriter  # for GIF export
from scipy.signal import savgol_filter

from typing import List, Union

import colors, helpers_builders


# def plot_arm(pos_vec: np.ndarray, buckle: np.ndarray, thetas: Union[np.ndarray, list, tuple], L: float, modality: str,
#              show: bool=True) -> None:
#     """
#     Plot bistable hinge chain (arm) given all joint positions and hinge buckling states.

#     Parameters
#     ----------
#     pos_vec : 2D np.ndarray, (N, 2), node coordinates
#     buckle : 1D np.ndarray, (N-1,), with hinge buckling orientation, +1 is shim going down
#     thetas : array-like of float, (N-1,), Hinge angles in radians
#     L : float, edge length, used to scale the arcs and plot limits.
#     modality : str, Plotting mode that sets the color scheme. Recognized values:
#                     - "measurement" : primary color (e.g., blue)
#                     - "update"      : secondary color (e.g., orange)
#     arc_scale : float, optional, Relative radius of the plotted hinge arcs in units of L. Default is 0.2.

#     Notes
#     -----
#     - The tip angle shown in the title is computed from the positions via
#       `helpers_builders._get_tip_angle` (in radians) and converted to degrees
#       only for display.
#     - Hinge arcs are drawn such that the sign of `buckle[i]` controls whether
#       the arc is visually clockwise or counter-clockwise: a value of -1 will
#       swap the order of the start/end angles relative to +1.
#     """
#     colors_lst, red, custom_cmap = colors.color_scheme()
#     plt.rcParams["axes.prop_cycle"] = plt.cycler("color", colors_lst)

#     # Convert inputs to NumPy arrays for plotting
#     pos = np.asarray(pos_vec, dtype=float)
#     buckle_arr = np.asarray(buckle)
#     thetas_rad = np.asarray(thetas, dtype=float)  # radians

#     # Extract x, y and tip angle (converted to degrees only for display)
#     xs, ys = pos[:, 0], pos[:, 1]
#     tip_angle_deg = np.rad2deg(float(helpers_builders._get_tip_angle(pos_vec)))

#     # ---- figure ----
#     plt.figure(figsize=(4, 4))

#     if modality == "measurement" or modality == "desired":
#         clr = colors_lst[0]
#     elif modality == "update":
#         clr = colors_lst[2]
#     else:
#         # Fallback color if modality is unknown
#         clr = colors_lst[1]

#     # ------ edges and nodes ------
#     # Plot polyline (all links)
#     plt.plot(xs, ys, linewidth=4, color=clr)
#     # Scatter joints
#     plt.scatter(xs, ys, s=60, zorder=3, color=clr)
#     # Origin in black
#     plt.scatter([0], [0], s=60, zorder=3, color="k")

#     # ------ line of wall ------
#     plt.plot([xs[-1], xs[-1]],  # vertical line at tip x
#              [ys[-1] + 0.4 * L, ys[-1] - 0.4 * L],  # short segment
#              linestyle=":", color="k", linewidth=3.0)

#     # ------ buckle ------
#     diffs = pos_vec[2:, :]-pos_vec[:-2, :]
#     diffs_3d = np.concatenate((diffs, np.zeros((np.shape(diffs)[0], 1))), axis=1)
#     buckle_3d = np.concatenate((np.zeros((np.shape(buckle)[0], 2)), buckle), axis=1)
#     V_3d = np.cross(diffs_3d, buckle_3d)
#     V = V_3d[:, :2]
#     print(V)
#     ax = plt.gca()
#     for p, v in zip(pos_vec[1:-1], V):
#         arrow = patches.FancyArrowPatch(p, p + v*0.25, arrowstyle='-|>', mutation_scale=25, linewidth=3, capstyle='round',
#                                         joinstyle='round')
#         ax.add_patch(arrow)

#     # annotate tip
#     plt.annotate(f"({xs[-1]:.2f}, {ys[-1]:.2f}, {tip_angle_deg:.2f})",
#                  xy=(xs[-1], ys[-1]), xytext=(xs[-1] - 0.05, ys[-1] - 0.05))

#     # aesthetics
#     plt.axis("equal")
#     plt.xlim(xs.min() - 0.5 * L, xs.max() + 0.5 * L)
#     plt.ylim(ys.min() - 0.5 * L, ys.max() + 0.5 * L)
#     plt.xlabel("x")
#     plt.ylabel("y")
#     if modality is not None:
#         plt.title(modality)
#     else:    
#         plt.title(f"Tip (x, y, theta)=({xs[-1]:.2f}, {ys[-1]:.2f}, {tip_angle_deg:.2f})")
    
#     if show:
#         plt.show()


def plot_arm(pos_vec: np.ndarray, buckle: np.ndarray, thetas: Union[np.ndarray, list, tuple], L: float,
             modality: str, show: bool=True, ax=None) -> None:
    colors_lst, red, custom_cmap = colors.color_scheme()

    # pick axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.figure

    xs, ys = pos_vec[:, 0], pos_vec[:, 1]
    tip_angle_deg = np.rad2deg(float(helpers_builders._get_tip_angle(pos_vec)))

    if modality in {"measurement", "desired"}:
        clr = colors_lst[0]
    elif modality == "update":
        clr = colors_lst[2]
    else:
        clr = colors_lst[1]

    # ------ edges and nodes ------
    ax.plot(xs, ys, linewidth=4, color=clr)
    ax.scatter(xs, ys, s=60, zorder=3, color=clr)
    ax.scatter([0], [0], s=60, zorder=3, color="k")

    # ------ line of wall ------
    ax.plot([xs[-1], xs[-1]],
            [ys[-1] + 0.4 * L, ys[-1] - 0.4 * L],
            linestyle=":", color="k", linewidth=3.0)

    # ------ buckle arrows ------
    diffs = pos_vec[2:, :] - pos_vec[:-2, :]
    diffs_3d = np.concatenate((diffs, np.zeros((diffs.shape[0], 1))), axis=1)
    buckle_3d = np.concatenate((np.zeros((buckle.shape[0], 2)), buckle), axis=1)
    V_3d = np.cross(diffs_3d, buckle_3d)
    V = V_3d[:, :2]

    for p, v in zip(pos_vec[1:-1], V):
        arrow = patches.FancyArrowPatch(
            p, p + v/np.linalg.norm(v)*0.02,
            arrowstyle='-|>',
            mutation_scale=25,
            linewidth=3,
            capstyle='round',
            joinstyle='round'
        )
        ax.add_patch(arrow)

    # annotate tip
    ax.annotate(rf"$x={xs[-1]:.2f},\ y={ys[-1]:.2f},\ \theta={tip_angle_deg:.2f}$",
                xy=(xs[-1], ys[-1]), xytext=(xs[-1] - 0.05, ys[-1] - 0.05))

    # aesthetics
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlim(xs.min() - 0.5 * L, xs.max() + 0.5 * L)
    ax.set_ylim(ys.min() - 0.5 * L, ys.max() + 0.5 * L)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(modality if modality is not None
                 else f"Tip (x, y, theta)=({xs[-1]:.2f}, {ys[-1]:.2f}, {tip_angle_deg:.2f})")

    if show and ax is None:
        plt.show()


def animate_arm_w_arcs(traj_pos, L, frames=10, interval_ms=30, save_path=None, fps=30, show_inline=False, buckle_traj=None,
                       theta_traj=None, arc_scale: float = 0.2):
    """
    Animate an N-link arm over time, optionally drawing hinge arcs.

    traj_pos   : array-like, shape (T, N, 2), positions over time
    L          : reference link length
    buckle_traj: optional, shape (T, H, S) or (T, H), buckle states per frame
    theta_traj : optional, shape (T, H), hinge angles per frame [rad]
    """
    colors_lst, red, custom_cmap = colors.color_scheme()
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", colors_lst)

    pos = np.asarray(traj_pos, dtype=float)  # (T, N, 2)
    T_all = pos.shape[0]
    assert pos.ndim == 3 and pos.shape[2] == 2

    # --- downsample time ---
    stride = max(1, int(T_all / frames))
    pos = pos[::stride]
    T, N, _ = pos.shape

    if buckle_traj is not None:
        buckle_traj = np.asarray(buckle_traj)[::stride]  # (T, H, S) or (T, H)
    if theta_traj is not None:
        theta_traj = np.asarray(theta_traj)[::stride]    # (T, H)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim([-L, 8 * L])
    ax.set_ylim([-4.5 * L, 4.5 * L])
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Polyline + joints + tip label
    (line,) = ax.plot([], [], linewidth=4)
    scat = ax.scatter([], [], s=60, zorder=3)
    tip_text = ax.text(0, 0, "", va="bottom", ha="left")

    # List to hold current arc patches so we can remove them each frame
    arc_patches: list[patches.Arc] = []
    r = arc_scale * float(L)

    def init():
        line.set_data([], [])
        scat.set_offsets(np.empty((0, 2)))
        tip_text.set_text("")
        # clear any leftover arcs
        for a in arc_patches:
            a.remove()
        arc_patches.clear()
        return line, scat, tip_text

    def update(ti):
        pts = pos[ti]  # (N, 2)
        xs, ys = pts[:, 0], pts[:, 1]
        line.set_data(xs, ys)
        scat.set_offsets(pts)
        tip_text.set_position((xs[-1], ys[-1]))
        tip_text.set_text(f"Tip ({xs[-1]:.2f}, {ys[-1]:.2f})")
        ax.set_title(f"update, t= {ti + 1}/{T}")

        # ---- remove previous arcs ----
        for a in arc_patches:
            a.remove()
        arc_patches.clear()

        # ---- draw hinge arcs if data provided ----
        if buckle_traj is not None and theta_traj is not None:
            buckle = np.asarray(buckle_traj[ti])
            # thetas_rad = np.asarray(theta_traj[ti], dtype=float)  # (H,)
            # buckle_arr = buckle  # possibly (H, S) with S=1

            # # cumulative hinge angles in radians
            # cumsum_thetas = np.cumsum(thetas_rad)
            # # shifted by pi but still in radians
            # cumsum_thetas_shift = cumsum_thetas - cumsum_thetas[0] + np.pi

            # for i in range(buckle_arr.shape[0]):
            #     # handle HxS or H arrays
            #     b_i = buckle_arr[i]
            #     # if 2D, take the first shim
            #     if np.ndim(b_i) > 0:
            #         b_i = b_i[0]

            #     p = pts[i + 1]  # hinge at node i+1

            #     if b_i == -1:
            #         theta1 = cumsum_thetas[i]
            #         theta2 = cumsum_thetas_shift[i]
            #     else:
            #         theta1 = cumsum_thetas_shift[i]
            #         theta2 = cumsum_thetas[i]

            #     theta1_deg = float(np.rad2deg(theta1))
            #     theta2_deg = float(np.rad2deg(theta2))

            #     arc = patches.Arc(xy=(p[0], p[1]), width=2 * r, height=2 * r, angle=0.0, theta1=theta1_deg, theta2=theta2_deg,
            #                       linewidth=2, zorder=2)
            #     ax.add_patch(arc)
            #     arc_patches.append(arc)

            diffs = pts[2:, :]-pts[:-2, :]
            diffs_3d = np.concatenate((diffs, np.zeros((np.shape(diffs)[0], 1))), axis=1)
            buckle_3d = np.concatenate((np.zeros((np.shape(buckle)[0], 2)), buckle), axis=1)
            V_3d = np.cross(diffs_3d, buckle_3d)
            V = V_3d[:, :2]
            # ax = plt.gca()
            for p, v in zip(pts[1:-1], V):
                arrow = patches.FancyArrowPatch(p, p + v/np.linalg.norm(v)*0.035, arrowstyle='-|>', mutation_scale=25, linewidth=2, capstyle='round',
                                                joinstyle='round')
                ax.add_patch(arrow)
                arc_patches.append(arrow)

        return line, scat, tip_text, *arc_patches

    anim = FuncAnimation(
        fig,
        update,
        frames=T,
        init_func=init,
        interval=interval_ms,
        blit=True,
    )

    if save_path is not None:
        if save_path.lower().endswith(".gif"):
            anim.save(save_path, writer=PillowWriter(fps=fps))
        elif save_path.lower().endswith(".mp4"):
            anim.save(save_path, writer="ffmpeg", fps=fps)
        else:
            raise ValueError("save_path must end with .gif or .mp4")

    if show_inline:
        from IPython.display import HTML

        return HTML(anim.to_jshtml())

    plt.close(fig)
    return fig, anim


# def loss_and_buckle_in_t(loss_in_t, buckle_in_t, start=0, end=None):
#     """
#     plot the SE loss as well as the buckle a.f.o simulation time.

#     loss_in_t  : np array of (T, 2), loss vector in time
#     buckle_in_t: np array of (T, H), hinge buckles in time
#     start      : optional, time to start plot
#     end        : optional, time to end plot, if simulation cesses before total simulation time
#     """
#     # -------- time vector --------
#     if end is None:
#         end = np.shape(loss_in_t)[1]

#     t = np.arange(start-1, end)   # integer indices

#     # -------- instantiate plot --------
#     fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=False)

#     # -------- subplot 1: loss --------
#     # axes[0].plot(t, np.sum(np.sqrt(loss_in_t[start-1:end, :]**2), axis=1))
#     axes[0].plot(t, np.sum(np.sqrt(loss_in_t[start:end+1, :]**2), axis=1))
#     axes[0].set_ylabel("Loss")
#     # integer ticks only, auto-spaced
#     axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

#     # -------- subplot 2: buckle states --------
#     H = np.shape(buckle_in_t)[0]
#     for i in range(H):
#         axes[1].plot(t, buckle_in_t[i, 0, start-1:end], label=f"hinge {i+1}")

#     # -------- beautify --------
#     axes[1].set_ylabel("buckle")
#     axes[1].set_xlabel("training step")
#     axes[1].legend()

#     axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

#     plt.tight_layout()
#     plt.show()

def loss_and_buckle_in_t(loss_MSE_in_t, buckle_in_t, F_meas_in_t, F_des_in_t, start=0, end=None):
    """
    Plot (top->bottom):
      1) measured forces (solid) vs desired forces (dotted) over training steps
      2) SE loss over training steps
      3) buckle states over training steps

    Parameters
    ----------
    loss_in_t    : np.ndarray, shape (T, 2)
    buckle_in_t  : np.ndarray, shape (H, 1, T)  (as in your current indexing)
    F_meas_in_t  : np.ndarray, shape (T, 2)
    F_des_in_t   : np.ndarray, shape (T, 2)
    start        : int, inclusive
    end          : int, exclusive (None -> full length)
    """
    # ------ colors ------
    colors_lst, red, custom_cmap = colors.color_scheme()

    # -------- time vector / slicing --------
    T = np.size(loss_MSE_in_t)
    if end is None or end > T:
        end = T
    if start < 0:
        start = 0

    t = np.arange(start, end)

    # -------- instantiate plot --------
    fig, axes = plt.subplots(3, 1, figsize=(6, 9), sharex=True)

    # -------- subplot 0: forces --------
    # measured (solid)
    axes[0].plot(t, F_meas_in_t[0, start:end]/1000, color=colors_lst[1], linestyle='-', label=r"$F_x$ meas")
    axes[0].plot(t, F_meas_in_t[1, start:end]/1000, color=colors_lst[2], linestyle='-', label=r"$F_y$ meas")

    # desired (dotted)
    axes[0].plot(t, F_des_in_t[0, start:end]/1000, color=colors_lst[1], linestyle=':', label=r"$F_x$ des")
    axes[0].plot(t, F_des_in_t[1, start:end]/1000, color=colors_lst[2], linestyle=':', label=r"$F_y$ des")

    axes[0].set_ylabel("Force [N]")
    axes[0].legend(ncol=2)
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # -------- subplot 1: loss --------
    axes[1].plot(t, loss_MSE_in_t[start:end])
    axes[1].set_ylabel("Loss")
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    # -------- subplot 2: buckle states --------
    H = buckle_in_t.shape[0]
    for i in range(H):
        axes[2].plot(t, buckle_in_t[i, 0, start:end], label=f"hinge {i+1}")

    axes[2].set_ylabel("buckle")
    axes[2].set_xlabel("t")
    axes[2].legend()
    axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()


def plot_compare_sim_exp_stress_strain(exp_dfs: List[pd.DataFrame], sim_df: pd.DataFrame, translate_ratio: float) -> None:
    """
    Plot experimental and simulated stress–strain curves for comparison of a full chain sumulation.

    Parameters
    ----------
    exp_dfs : List[pandas.DataFrame]
        A list of experimental dataframes. Each dataframe must contain
        the columns:
            - "Position (mm)" : tip position in millimeters
            - "Load2 (N)"    : measured load (force) in Newtons

    sim_df : pandas.DataFrame
        Simulation results. Must contain:
            - "x_tip" : simulated tip x-position
            - "Fx"    : simulated x-direction force

    translate_ratio : float
        Factor converting displacement units (e.g., mm). Applied as:
            (x_tip - x_tip_initial) * translate_ratio

    Returns
    -------
    None
        matplotlib figure

    Notes
    -----
    - Experimental curves are smoothed using a Savitzky–Golay filter
      with window length 16 and polynomial order 4.
    - Simulation force is plotted as -Fx to match the experimental sign
      convention.
    """  
    colors_lst, red, custom_cmap = colors.color_scheme()
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", colors_lst) 
    font_size = 16
    
    # experimental
    window = 16
    for i, exp_df in enumerate(exp_dfs):
        exp_df_pos = exp_df["Position (mm)"]
        exp_df_load = exp_df["Load2 (N)"]
        exp_df_load_movmean = savgol_filter(exp_df_load, window_length=window, polyorder=4, mode="interp")
        # plt.plot(exp_df_pos, exp_df_load_movmean, linewidth=1.0, linestyle=":")
        plt.plot(exp_df_pos, exp_df_load_movmean, linewidth=1.0)

    # simulation - change to look like experiment
    # sim_tip = (sim_df['x_tip'] - sim_df['x_tip'][0]) / translate_ratio * 2.6
    sim_tip = (sim_df['x_tip'] - sim_df['x_tip'][0]) * translate_ratio
    # sim_Fx = -sim_df['Fx'] * 0.045
    sim_Fx = -sim_df['Fx']
    plt.plot(sim_tip, sim_Fx, '.', markersize=10.0, color=colors_lst[3])

    # Legend: experiment 1, experiment 2, ..., simulation
    legend_labels = [f"experiment {i+1}" for i in range(len(exp_dfs))]
    legend_labels.append("simulation")

    # Beautify
    plt.ylim([-0.15, 0.15])
    plt.xlabel("pos [mm]", fontsize=font_size)
    plt.ylabel("Force [N]", fontsize=font_size)
    plt.legend(legend_labels, fontsize=font_size)
    plt.show()


def plot_tau_afo_theta(torque_func) -> None:
    thetas = np.linspace(-np.pi, np.pi, 100)
    taus = torque_func(thetas)
    plt.plot(thetas, taus)
    plt.ylabel(r'$\tau$')
    plt.xlabel(r'$\theta\,\left[rad\right]$')
    plt.ylim([-15, 15])
    plt.show()


# # ==========
# # NOT IN USE
# # ==========

# def plot_energies(Variabs: "VariablesClass", Strctr: "StructureClass", pos_in_t: np.array[np.float_], Energy_func, ):
#     T = np.shape(pos_in_t)[0]
#     energies = np.zeros(int(T))
#     for i in range(int(T)):
#         energies[i], _, _ = Energy_func(Variabs, Strctr, pos_in_t[i])
        
#     plt.plot(energies)
#     plt.yscale('log')

# def animate_arm(traj_pos, L, frames=10, interval_ms=30, save_path=None, fps=30, show_inline=False):
#     """
#     Animate an N-link arm over time.
#     traj_pos: array-like, shape (T, N, 2), positions over time
#     L: reference link length (used only for nice padding if needed)
#     interval_ms: delay between frames (for interactive playback)
#     save_path: if provided, writes an animation ('.gif' or '.mp4')
#     fps: frames per second when saving

#     Returns: (fig, anim) so you can display or save later.
#     """
#     pos = np.asarray(traj_pos)              # (T, N, 2)
#     T = np.shape(pos)[0]
#     assert pos.ndim == 3 and pos.shape[2] == 2

#     # --- downsample time ---
#     stride = int(T/frames)
#     pos = pos[::max(1, int(stride))]
#     T, N, _ = pos.shape

#     fig, ax = plt.subplots(figsize=(4, 4))
#     ax.set_aspect('equal', adjustable='box')
#     ax.set_xlim([-L, 8*L])
#     ax.set_ylim([-4.5*L, 4.5*L])
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")

#     # Polyline + joints + tip label
#     line, = ax.plot([], [], linewidth=4)
#     scat = ax.scatter([], [], s=60, zorder=3)
#     tip_text = ax.text(0, 0, "", va="bottom", ha="left")

#     def init():
#         line.set_data([], [])
#         scat.set_offsets(np.empty((0, 2)))
#         tip_text.set_text("")
#         return line, scat, tip_text

#     def update(ti):
#         pts = pos[ti]                    # (N, 2)
#         xs, ys = pts[:, 0], pts[:, 1]
#         line.set_data(xs, ys)
#         scat.set_offsets(pts)
#         tip_text.set_position((xs[-1], ys[-1]))
#         tip_text.set_text(f"Tip ({xs[-1]:.2f}, {ys[-1]:.2f})")
#         ax.set_title(f"Frame {ti+1}/{T}")
#         return line, scat, tip_text

#     anim = FuncAnimation(fig, update, frames=T, init_func=init,
#                          interval=interval_ms, blit=True)

#     if save_path is not None:
#         if save_path.lower().endswith(".gif"):
#             anim.save(save_path, writer=PillowWriter(fps=fps))
#         elif save_path.lower().endswith(".mp4"):
#             # Requires ffmpeg installed
#             anim.save(save_path, writer="ffmpeg", fps=fps)
#         else:
#             raise ValueError("save_path must end with .gif or .mp4")

#     # ---- Inline display (keep small!) ----
#     if show_inline:
#         # reduce embed size by downsampling and smaller fig/dpi
#         from IPython.display import HTML
#         return HTML(anim.to_jshtml())

#     plt.close(fig)
#     return fig, anim

