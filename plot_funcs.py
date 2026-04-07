import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from IPython.display import HTML
from matplotlib import patches
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation, PillowWriter  # for GIF export
from matplotlib.colors import BoundaryNorm
from scipy.signal import savgol_filter
from matplotlib.patches import Ellipse, FancyArrowPatch
from collections import Counter

from typing import Tuple, List, Union
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional

import colors, helpers_builders


# -------------------------------------------------
# Plot importants
# -------------------------------------------------
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


def loss_and_buckle_in_t(tip_pos_in_t, tip_angle_in_t, loss_in_t, buckle_in_t, F_meas_in_t, F_des_in_t,
                         tip_pos_update_in_t, tip_angle_update_in_t, start=0, end=None,
                         save_path: Optional[str] = None) -> None:
    """
    Plot (top->bottom):
      1) measured forces (solid) vs desired forces (dotted) over training steps
      2) SE loss over training steps
      3) buckle states over training steps

    Parameters
    ----------
    loss_in_t - np.ndarray, shape (T, 2)
    buckle_in_t   - np.ndarray, shape (H, 1, T)  (as in your current indexing)
    F_meas_in_t   - np.ndarray, shape (T, 2)
    F_des_in_t    - np.ndarray, shape (T, 2)
    start         - int, inclusive
    end           - int, exclusive (None -> full length)
    """
    # ------ colors ------
    colors_lst, _, _ = colors.color_scheme()

    # -------- time vector / slicing --------
    T = np.shape(loss_in_t)[0]
    if end is None or end > T:
        end = T
    if start < 0:
        start = 0

    t = np.arange(start, end)

    # -------- instantiate plot --------
    fig, axes = plt.subplots(5, 1, figsize=(6, 9), sharex=True)

    # -------- subplot 0: positions --------
    # axes[0].plot(t, tip_pos_in_t[start:end, 0], color=colors_lst[1], linestyle='-', label=r"$tip_x$ meas")
    # axes[0].plot(t, tip_pos_in_t[start:end, 1], color=colors_lst[2], linestyle='-', label=r"$tip_y$ meas")
    # axes[0].plot(t, tip_angle_in_t[start:end], color=colors_lst[3], label=r"$\theta$ meas")

    # # dashed at 0
    # axes[0].plot(t, np.zeros(end-start), color='k', linestyle='--')

    # axes[0].set_ylabel("pos [mm]")
    # axes[0].legend(ncol=2)
    # axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    ax = axes[0]

    # ---- left axis: position ----
    ax.plot(t, tip_pos_in_t[start:end, 1], color=colors_lst[2], linestyle='-', label=r"$tip_y$ meas")

    ax.set_ylabel(r"$tip_y\left[mm\right]$")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ---- right axis: angle ----
    ax2 = ax.twinx()

    ax2.plot(t, tip_angle_in_t[start:end],
             color=colors_lst[3], label=r"$\theta$ meas")

    ax2.set_ylabel(r"$\theta\left[rad\right]$")

    # ---- combined legend ----
    lines = ax.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, ncol=2)

    # dashed at 0
    ax.plot(t, np.zeros(end-start), color='k', linestyle='--')

    # -------- subplot 1: positions --------
    # measured (solid)
    axes[1].plot(t, F_meas_in_t[0, start+1:end+1], color=colors_lst[1], linestyle='-', label=r"$F_x$ meas")
    axes[1].plot(t, F_meas_in_t[1, start+1:end+1], color=colors_lst[2], linestyle='-', label=r"$F_y$ meas")

    # desired (dotted)
    axes[1].plot(t, F_des_in_t[0, start+1:end+1], color=colors_lst[1], linestyle=':', label=r"$F_x$ des")
    axes[1].plot(t, F_des_in_t[1, start+1:end+1], color=colors_lst[2], linestyle=':', label=r"$F_y$ des")

    # dashed at 0
    axes[1].plot(t, np.zeros(end-start), color='k', linestyle='--')

    axes[1].set_ylabel("Force [mN]")
    axes[1].legend(ncol=2)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].set_ylim([-200, 500])

    # -------- subplot 1: loss --------
    axes[2].plot(t, loss_in_t[start+1:end+1, 0], color=colors_lst[1])
    axes[2].plot(t, loss_in_t[start+1:end+1, 1], color=colors_lst[2])
    loss_MSE_in_t = np.sqrt(np.sum(loss_in_t**2, axis=1))
    axes[2].plot(t, loss_MSE_in_t[start+1:end+1])

    # dashed at 0
    axes[2].plot(t, np.zeros(end-start), color='k', linestyle='--')

    axes[2].set_ylabel("Loss")
    axes[2].legend([r'$L_x$', r'$L_y$', r'$\|L\|$'])
    axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[2].set_ylim([-2.0, 2.0])

    # ------ subplot 4: delta tip update ------
    ax3 = axes[3]
    ax3.plot(t[:], tip_pos_update_in_t[start+1:end+1, 0] - tip_pos_update_in_t[start:end, 0],
             label=r"$\Delta tip_x^{\,!}\left[mm\right]$")
    ax3.plot(t[:], tip_pos_update_in_t[start+1:end+1, 1] - tip_pos_update_in_t[start:end, 1],
             label=r"$\Delta tip_y^{\,!}\left[mm\right]$")
    ax3.set_ylabel(r"$\Delta tip^{\,!}\left[mm\right]$")
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ---- right axis: angle ----
    ax3_2 = ax3.twinx()
    ax3_2.plot(t[:], tip_angle_update_in_t[start+1:end+1] - tip_angle_update_in_t[start:end],
               color=colors_lst[3], label=r"$\Delta\theta^{\,!}\left[rad\right]$")
    ax3_2.set_ylabel(r"$\Delta\theta^{\,!}\left[rad\right]$")

    lines = ax3.get_lines() + ax3_2.get_lines()
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, ncol=2)

    # dashed at 0
    axes[3].plot(t, np.zeros(end-start), color='k', linestyle='--')

    # -------- subplot 2: buckle states --------
    H = buckle_in_t.shape[0]
    for i in range(H):
        axes[4].plot(t, buckle_in_t[i, 0, start:end], label=f"hinge {i+1}")

    axes[4].set_ylabel("buckle")
    axes[4].set_xlabel("t")
    axes[4].legend()
    axes[4].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)


# ------------------------------------------------
# Stress-strains
# ------------------------------------------------
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


# --------------------------------------------------------
# Animations
# --------------------------------------------------------
def animate_arm_w_arcs(traj_pos, L, frames=10, interval_ms=30, save_path=None, fps=30, buckle_traj=None):
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

    anim = FuncAnimation(fig, update, frames=T, init_func=init, interval=interval_ms, blit=True)

    if save_path is not None:
        if save_path.lower().endswith(".gif"):
            anim.save(save_path, writer=PillowWriter(fps=fps))
        elif save_path.lower().endswith(".mp4"):
            anim.save(save_path, writer="ffmpeg", fps=fps)
        else:
            raise ValueError("save_path must end with .gif or .mp4")

    plt.close(fig)
    return fig, anim


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

    M_masked = np.ma.masked_where(np.eye(M.shape[0], dtype=bool), M)

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


def plot_success_matrix_with_pathways(M_corr: np.ndarray, title: str = "Training success matrix (pathways corrected)"):
    """
    Codes:
    0 - direct success
    1 - missing
    2 - direct failure
    3 - indirect success via pathway
    """
    # direct success, missing, failure, indirect success
    colors_lst, _, custom_cmap = colors.color_scheme()
    # norm = BoundaryNorm([0, 1, 2, 3], custom_cmap.N)
    # IMPORTANT: order = index value
    # cmap = ListedColormap([
    #     colors_lst[1],  # 0 → direct success
    #     colors_lst[3],  # 1 → missing (even if hidden)
    #     colors_lst[2],  # 2 → failure
    #     colors_lst[0],  # 3 → pathway success
    # ])

    norm = BoundaryNorm([-0, 0.5, 1.5, 2.5, 3], custom_cmap.N)

    M_corr_masked = np.ma.masked_where(np.eye(M_corr.shape[0], dtype=bool), M_corr)

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    # ax.imshow(M_corr_masked[::-1, :], cmap=custom_cmap, norm=norm, interpolation="none", aspect="equal")
    ax.imshow(M_corr_masked[::-1, :], cmap=custom_cmap, norm=norm, interpolation="none", aspect="equal")

    labels = [helpers_builders.index_to_buckle(i) for i in range(16)]
    ax.set_xticks(np.arange(16))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(np.arange(16))
    ax.set_yticklabels(labels[::-1])

    ax.set_xlabel("desired buckle")
    ax.set_ylabel("initial buckle")
    ax.set_title(title)

    legend_handles = [
        patches.Patch(facecolor=custom_cmap(norm(0)), label="Direct success"),
        patches.Patch(facecolor=custom_cmap(norm(3)), label="Pathway success"),
        # patches.Patch(facecolor=colors_lst[3], label="Missing"),
        patches.Patch(facecolor=custom_cmap(norm(1)), label="Failure"),
    ]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.show()


def plot_transition_diagram(transitions: Counter, n_bits: int, *, only_reached_nodes: bool = False):
    colors_lst, _, _ = colors.color_scheme()
    node_edge = "black"
    node_face = "white"
    arrow_color = colors_lst[0]
    text_color = "black"

    pos = helpers_builders.state_positions(4)

    if only_reached_nodes:
        used_nodes = set()
        for a, b in transitions:
            used_nodes.add(a)
            used_nodes.add(b)
    else:
        used_nodes = set(helpers_builders.all_binary_states(n_bits))

    fig, ax = plt.subplots(figsize=(12, 8))

    # ---- nodes ----
    if only_reached_nodes:
        used_nodes = set()
        for a, b in transitions:
            used_nodes.add(a)
            used_nodes.add(b)
    else:
        used_nodes = set(helpers_builders.all_binary_states(4))

    for s in helpers_builders.all_binary_states(4):
        if s not in used_nodes:
            continue
        x, y = pos[s]
        node = Ellipse((x, y), width=0.1, height=0.08,
                       facecolor=node_face, edgecolor=node_edge, lw=2.5)
        ax.add_patch(node)
        ax.text(x, y, s, ha="center", va="center", fontsize=18, color=text_color)

    # ---- edges ----
    if transitions:
        max_count = max(transitions.values())
    else:
        max_count = 1

    for (src, dst), count in transitions.items():
        source = helpers_builders.index_to_buckle(src)
        dist = helpers_builders.index_to_buckle(dst)
        x1, y1 = pos[source]
        x2, y2 = pos[dist]

        # slight curvature if reverse edge also exists
        rev_exists = (dst, src) in transitions
        rad = 0.18 if rev_exists and src < dst else (-0.18 if rev_exists else 0.0)

        lw = 1.5 + 3.0 * count / max_count  # edge width, change to uniform for all arrows at same width

        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>",
            mutation_scale=16,
            lw=lw,
            color=arrow_color,
            shrinkA=28,
            shrinkB=28,
            connectionstyle=f"arc3,rad={rad}",
        )
        ax.add_patch(arrow)

        # if show_edge_labels:
        #     xm = 0.5 * (x1 + x2)
        #     ym = 0.5 * (y1 + y2)
        #     ax.text(xm, ym + 0.12, str(count), fontsize=10, ha="center", va="center")

    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


# # ==========
# # NOT IN USE
# # ==========

# def plot_energies(Variabs: "VariablesClass", Strctr: "StructureClass", pos_in_t: np.array[np.float64], Energy_func, ):
#     T = np.shape(pos_in_t)[0]
#     energies = np.zeros(int(T))
#     for i in range(int(T)):
#         energies[i], _, _ = Energy_func(Variabs, Strctr, pos_in_t[i])
        
#     plt.plot(energies)
#     plt.yscale('log')
