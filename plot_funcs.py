import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

from matplotlib import patches
from matplotlib.animation import FuncAnimation, PillowWriter  # for GIF export
from scipy.signal import savgol_filter

from typing import List, Union
from numpy.typing import NDArray

import colors, helpers_builders


def plot_arm(pos_vec: np.ndarray, buckle: np.ndarray, L: float, modality: str, show: bool = True, ax=None) -> None:
    """
    Plot arm configuration together with buckle direction arrows.

    Parameters
    ----------
    pos_vec   - ndarray, shape ``(nodes, 2)``, xy coordinates of chain nodes.
    buckle    - ndarray, shape ``(H,)`` or ``(H, 1)``, buckle sign of each hinge. 1=down, -1=up
    L         - float, characteristic link length used for visual scaling.
    modality  - Optional[str], selects chain color. ``"measurement"`` and ``"desired"`` use one color, ``"update"`` another.
    show      - bool, if True and ``ax`` not provided, display figure.
    ax        - Optional[Axes], existing matplotlib axes to draw on.
    """
    # ------ prelims ------
    colors_lst, _, _ = colors.color_scheme()

    # pick axes
    created_ax = ax is None
    if created_ax:
        _, ax = plt.subplots(figsize=(4, 4))

    xs, ys = pos_vec[:, 0], pos_vec[:, 1]
    tip_angle_deg = np.rad2deg(float(helpers_builders._get_tip_angle(pos_vec)))

    if modality in {"measurement", "desired"}:
        clr = colors_lst[0]
    elif modality == "update":
        clr = colors_lst[2]
    else:
        clr = colors_lst[1]

    # ------ chain faces and nodes ------
    ax.plot(xs, ys, linewidth=4, color=clr)
    ax.scatter(xs, ys, s=60, zorder=3, color=clr)
    ax.scatter([0], [0], s=60, zorder=3, color="k")

    # # ------ line of wall ------
    # ax.plot([xs[-1], xs[-1]],
    #         [ys[-1] + 0.4 * L, ys[-1] - 0.4 * L],
    #         linestyle=":", color="k", linewidth=3.0)

    # ------ buckle arrows ------
    buckle_vec = np.asarray(buckle, dtype=float).reshape(-1, 1)
    diffs = pos_vec[2:, :] - pos_vec[:-2, :]
    diffs_3d = np.column_stack((diffs, np.zeros(diffs.shape[0], dtype=float)))
    buckle_3d = np.column_stack((np.zeros((buckle_vec.shape[0], 2), dtype=float), buckle_vec.reshape(-1)))
    directions = np.cross(diffs_3d, buckle_3d)[:, :2]

    for p, v in zip(pos_vec[1:-1], directions):
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-12:
            continue

        arrow = patches.FancyArrowPatch(
            p,
            p + (v / norm_v) * 0.02,
            arrowstyle="-|>",
            mutation_scale=25,
            linewidth=3,
            capstyle="round",
            joinstyle="round",
        )
        try:
            ax.add_patch(arrow)
        except Exception:
            print("bad animation, lets solve this later")

    # ------ annotate tip and aesthetics -------
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

    if show and created_ax:
        plt.show()


def animate_arm_w_arcs(traj_pos, L, frames=10, interval_ms=30, save_path=None, fps=30, show_inline=False,
                       buckle_traj=None) -> None:
    """
    Animate an N-link arm over time, optionally drawing hinge arcs.

    Parameters
    ----------
    traj_pos    - array-like, shape ``(T, N, 2)``, arm positions over time.
    L           - float, reference link length used for axis scaling.
    frames      - int, approximate number of displayed frames after temporal downsampling.
    interval_ms - int, delay between displayed frames in milliseconds.
    save_path   - Optional[str], if given, save the animation to ``.gif`` or ``.mp4``.
    fps         - int, output frame rate used when saving.
    show_inline - bool, if True, return an HTML animation object for notebook display.
    buckle_traj - Optional[array-like], buckle history with shape ``(T, H, S)`` or static buckle state ``(H, S)``.
    arc_scale   - float, kept for interface compatibility. Currently unused.

    Returns
    -------
    (fig, anim) - tuple[Figure, FuncAnimation], returned when ``show_inline=False``.
    HTML        - IPython display object, returned when ``show_inline=True``.
    """
    colors_lst, _, _ = colors.color_scheme()
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", colors_lst)

    pos = np.asarray(traj_pos, dtype=float)  # (T, N, 2)
    T_all = pos.shape[0]
    assert pos.ndim == 3 and pos.shape[2] == 2

    if np.shape(buckle_traj)[0] != np.shape(traj_pos)[0]:
        buckle_traj = np.tile(buckle_traj, np.shape(traj_pos)[0]).T.reshape(np.shape(traj_pos)[0],
                                                                            np.shape(traj_pos)[1]-2, 1)

    # --- downsample time ---
    stride = max(1, int(T_all / frames))
    pos = pos[::stride]
    T, N, _ = pos.shape

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
        ax.set_title(f"t= {ti + 1}/{T}")

        # ---- remove previous arcs ----
        for a in arc_patches:
            a.remove()
        arc_patches.clear()

        # ---- draw hinge arcs if data provided ----
        if buckle_traj is not None:
            buckle = np.asarray(buckle_traj[ti])

            diffs = pts[2:, :]-pts[:-2, :]
            diffs_3d = np.concatenate((diffs, np.zeros((np.shape(diffs)[0], 1))), axis=1)
            buckle_3d = np.concatenate((np.zeros((np.shape(buckle)[0], 2)), buckle), axis=1)
            V_3d = np.cross(diffs_3d, buckle_3d)
            V = V_3d[:, :2]
            for p, v in zip(pts[1:-1], V):
                arrow = patches.FancyArrowPatch(p, p + v/np.linalg.norm(v)*0.035, arrowstyle='-|>', mutation_scale=25, 
                                                linewidth=2, capstyle='round', joinstyle='round')
                try:
                    ax.add_patch(arrow)
                    arc_patches.append(arrow)
                except:
                    print('bad animation, lets solve this later')

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


def loss_and_buckle_in_t(loss_MSE_in_t, buckle_in_t, F_meas_in_t, F_des_in_t, start=0, end=None, save_path: str = None) -> None:
    """
    Plot (top->bottom):
      1) measured forces (solid) vs desired forces (dotted) over training steps
      2) SE loss over training steps
      3) buckle states over training steps

    Parameters
    ----------
    loss_MSE_in_t - np.ndarray, shape (T, 2)
    buckle_in_t   - np.ndarray, shape (H, 1, T)  (as in your current indexing)
    F_meas_in_t   - np.ndarray, shape (T, 2)
    F_des_in_t    - np.ndarray, shape (T, 2)
    start         - int, inclusive
    end           - int, exclusive (None -> full length)
    """
    # ------ colors ------
    colors_lst, _, _ = colors.color_scheme()

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
    axes[0].plot(t, F_meas_in_t[0, start:end], color=colors_lst[1], linestyle='-', label=r"$F_x$ meas")
    axes[0].plot(t, F_meas_in_t[1, start:end], color=colors_lst[2], linestyle='-', label=r"$F_y$ meas")

    # desired (dotted)
    axes[0].plot(t, F_des_in_t[0, start:end], color=colors_lst[1], linestyle=':', label=r"$F_x$ des")
    axes[0].plot(t, F_des_in_t[1, start:end], color=colors_lst[2], linestyle=':', label=r"$F_y$ des")

    axes[0].set_ylabel("Force [mN]")
    axes[0].legend(ncol=2)
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_ylim([-200, 500])

    # -------- subplot 1: loss --------
    axes[1].plot(t, loss_MSE_in_t[start:end])
    axes[1].set_ylabel("Loss")
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].set_ylim([-0.02, 6.2])

    # -------- subplot 2: buckle states --------
    H = buckle_in_t.shape[0]
    for i in range(H):
        axes[2].plot(t, buckle_in_t[i, 0, start:end], label=f"hinge {i+1}")

    axes[2].set_ylabel("buckle")
    axes[2].set_xlabel("t")
    axes[2].legend()
    axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_tau_afo_theta(torque_func) -> None:
    """
    Torque-angle response over ``[-pi, pi]``. Used inside main.ipynb for single hinge stress response

    Parameters
    ----------
    torque_func - callable, function mapping angle in radians to torque.
    """
    thetas = np.linspace(-np.pi, np.pi, 100)
    taus = torque_func(thetas)
    plt.plot(thetas, taus)
    plt.ylabel(r'$\tau$')
    plt.xlabel(r'$\theta\,\left[rad\right]$')
    plt.ylim([-15, 15])
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


# ----------------------------
# Post Processing
# ----------------------------
def plot_success_matrix(M: NDArray):

    colors_lst, _, custom_cmap = colors.color_scheme()

    labels = []
    for i in range(16):
        b = format(i, "04b")
        labels.append(b)

    plt.figure(figsize=(5, 5))

    M_masked = np.ma.masked_where(np.triu(np.ones_like(M), k=0), M)

    im = plt.imshow(M_masked, cmap=custom_cmap, vmin=0, vmax=4, origin="lower")
    # im = plt.imshow(M, cmap=custom_cmap, vmin=0, vmax=4, origin="lower")

    plt.xticks(range(16), labels, rotation=90)
    plt.yticks(range(16), labels)

    plt.xlabel("desired buckle")
    plt.ylabel("initial buckle")

    plt.title("Training success matrix")

    # plt.colorbar(label="success")
    legend_elements = [patches.Patch(facecolor=custom_cmap(im.norm(0)), label="Success"),
                       patches.Patch(facecolor=custom_cmap(im.norm(1)), label="Missing"),
                       patches.Patch(facecolor=custom_cmap(im.norm(2)), label="Failure")]

    plt.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
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
