import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from IPython.display import HTML
from matplotlib import patches
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation, PillowWriter  # for GIF export
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter
from matplotlib.patches import Ellipse, FancyArrowPatch
from collections import Counter
from pathlib import Path

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

    N_nodes = pos_vec[:, 0].shape[0]

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

        arrow = patches.FancyArrowPatch(p, p + (v / norm_v) * 0.004 * N_nodes, arrowstyle="-|>", mutation_scale=25,
                                        linewidth=3, capstyle="round", joinstyle="round")
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


def _buckle_hinge_first(buckle_in_t: NDArray, T: int) -> NDArray:
    """
    Return buckle history as ``(H, S, T)`` for plotting.

    Accepts both the historical plotting layout ``(H, S, T)`` and the
    time-first layout ``(T, H, S)`` used by imported training files and
    animations.
    """
    buckle = np.asarray(buckle_in_t)

    if buckle.ndim == 3:
        if buckle.shape[-1] == T:
            return buckle
        if buckle.shape[0] == T:
            return np.moveaxis(buckle, 0, -1)

    if buckle.ndim == 2:
        if buckle.shape[0] == T:
            return buckle.T[:, np.newaxis, :]
        if buckle.shape[1] == T:
            return buckle[:, np.newaxis, :]

    raise ValueError(
        "buckle_in_t must have time axis of length T and shape (H, S, T), "
        "(T, H, S), (H, T), or (T, H)"
    )


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
    loss_in_t   - np.ndarray, shape (T, 2)
    buckle_in_t - np.ndarray, shape (H, 1, T) or (T, H, 1)
    F_meas_in_t - np.ndarray, shape (T, 2)
    F_des_in_t  - np.ndarray, shape (T, 2)
    start       - int, inclusive
    end         - int, exclusive (None -> full length)
    """
    # ------ colors ------
    colors_lst, _, _ = colors.color_scheme()

    # -------- time vector / slicing and buckles --------
    T = np.shape(loss_in_t)[0]
    if end is None or end > T:
        end = T
    if start < 0:
        start = 0

    t = np.arange(start, end)

    buckle_in_t = _buckle_hinge_first(buckle_in_t, T)
    H = buckle_in_t.shape[0]

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
    if H < 6:
        axes[1].set_ylim([-200, 500])
    else:
        axes[1].set_ylim([-160, 160])

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
    for i in range(H):
        axes[4].plot(t, buckle_in_t[i, 0, start:end], label=f"hinge {i+1}")

    axes[4].set_ylabel("buckle")
    axes[4].set_xlabel("t")
    if H < 6:
        axes[4].legend()
    axes[4].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def buckle_state_colormap(n_bits: int) -> tuple[ListedColormap, BoundaryNorm]:
    """
    Build a stable buckle-state colormap.

    States are colored by their binary index, so the same buckle gets the same
    color across different sweep runs. For a 4-hinge chain this gives 15 colors
    sampled from the custom colormap and the 16th color as the custom red.
    """
    _, red, custom_cmap = colors.color_scheme()
    n_states = 2 ** int(n_bits)

    if n_states <= 1:
        palette = [red]
    else:
        palette = [custom_cmap(x) for x in np.linspace(0.0, 1.0, n_states - 1)]
        palette.append(red)

    cmap = ListedColormap(palette, name=f"buckle_state_{n_bits}bit")
    norm = BoundaryNorm(np.arange(-0.5, n_states + 0.5, 1.0), cmap.N)
    return cmap, norm


def plot_tip_grid_buckle_ids(buckle_grid_frames: NDArray, y_num: int, theta_num: int,
                             theta_min: float, theta_max: float, y_min: float, y_max: float, *,
                             grid_start: int = 1, snake: bool = True, y_scale: float = 1000.0,
                             save_path: Optional[str | Path] = None, show: bool = True, ax=None
                             ) -> tuple[plt.Figure, plt.Axes, NDArray[np.int32]]:
    """
    Plot the final buckle state reached at each y/theta grid point.

    Buckle colors are assigned by global binary buckle index, not by order of
    appearance in the current sweep, so colors remain stable across initial
    condition loops.
    """
    frames = np.asarray(buckle_grid_frames, dtype=int)
    n_bits = int(np.prod(frames.shape[1:]))

    grid_frames = frames[grid_start:]
    if grid_frames.shape[0] != y_num * theta_num:
        raise ValueError(
            f"Expected {y_num * theta_num} grid frames after grid_start={grid_start}, "
            f"got {grid_frames.shape[0]}."
        )

    buckle_ids = np.array(
        [helpers_builders.buckle_to_index(frame.reshape(-1)) for frame in grid_frames],
        dtype=np.int32,
    ).reshape(y_num, theta_num)

    if snake:
        buckle_ids[1::2, :] = buckle_ids[1::2, ::-1]

    cmap, norm = buckle_state_colormap(n_bits)
    extent = [theta_min, theta_max, y_min * y_scale, y_max * y_scale]

    created_ax = ax is None
    if created_ax:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    ax.imshow(buckle_ids, origin="lower", aspect="auto", extent=extent, cmap=cmap, norm=norm)
    ax.set_xlabel("tip angle [rad]")
    ax.set_ylabel("tip y [mm]")
    ax.set_title("Buckle state after update sweep")

    observed_ids = sorted(int(idx) for idx in np.unique(buckle_ids))
    handles = [
        patches.Patch(facecolor=cmap(norm(idx)), label=helpers_builders.index_to_buckle(idx, n_bits=n_bits))
        for idx in observed_ids
    ]
    if handles:
        ax.legend(handles=handles, title="buckle", bbox_to_anchor=(1.02, 1), loc="upper left")

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show and created_ax:
        plt.show()

    return fig, ax, buckle_ids


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


def plot_compare_sim_exp_stress_strain(exp_dfs: List[pd.DataFrame], sim_df: pd.DataFrame,
                                       translate_ratio: float) -> None:
    """
    Plot experimental and simulated stress–strain curves for comparison of a full chain simulation.

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


def plot_compare_sim_exp_training(exp_file_path: str, sim_file_path: str,
                                  translate_ratio: float, final_t: Optional[int] = None,
                                  save: bool = False) -> None:
    """
    Plot experimental and simulated training for comparison of a full chain simulation.

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

    # read experimental dataframe and extract sizes
    exp_df = pd.read_csv(exp_file_path)

    F_exp_meas = np.vstack([exp_df["F_x_meas"].to_numpy(dtype=float),
                            exp_df["F_y_meas"].to_numpy(dtype=float)])  # shape (2, T)
    F_exp_des = np.vstack([exp_df["F_x_des"].to_numpy(dtype=float),
                           exp_df["F_y_des"].to_numpy(dtype=float)])  # shape (2, T)
    loss_MSE_exp = exp_df["loss_MSE"].to_numpy(dtype=float)

    # read simulation dataframe
    sim_df = pd.read_csv(sim_file_path)

    F_sim_meas = np.vstack([sim_df["F_x_meas"].to_numpy(dtype=float),
                            sim_df["F_y_meas"].to_numpy(dtype=float)])  # shape (2, T)
    F_sim_des = np.vstack([sim_df["F_x_des"].to_numpy(dtype=float),
                           sim_df["F_y_des"].to_numpy(dtype=float)])  # shape (2, T)
    loss_MSE_sim = sim_df["loss_MSE"].to_numpy(dtype=float)

    # time steps
    T = int(F_exp_meas.shape[1])
    if final_t is None:
        final_t = T
    sl = slice(1, final_t)
    t = np.arange(T, dtype=int)

    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(6, 6), gridspec_kw={"height_ratios": [1, 1]})

    # ===== top: forces =====
    markersize = 10.0

    axs[0].plot(t[sl], F_exp_meas[0, sl],
                marker=".", linestyle="None", markersize=markersize,
                color=colors_lst[1], label="Fx measured")
    axs[0].plot(t[sl], F_exp_des[0, sl],
                marker=".", linestyle="None", markersize=markersize,
                color=colors_lst[1], label="Fx desired")

    axs[0].plot(t[sl], F_exp_meas[1, sl],
                marker=".", linestyle="None", markersize=markersize,
                color=colors_lst[2], label="Fy measured")
    axs[0].plot(t[sl], F_exp_des[1, sl],
                marker=".", linestyle="None", markersize=markersize,
                color=colors_lst[2], label="Fy desired")

    axs[0].plot(t[sl], F_sim_meas[0, sl], color=colors_lst[1], label="Fx measured")
    axs[0].plot(t[sl], F_sim_des[0, sl], color=colors_lst[1], linestyle="--", label="Fx desired")

    axs[0].plot(t[sl], F_sim_meas[1, sl], color=colors_lst[2], label="Fy measured")
    axs[0].plot(t[sl], F_sim_des[1, sl], color=colors_lst[2], linestyle="--", label="Fy desired")

    # ===== bottom: MSE loss =====
    axs[1].plot(t[sl], loss_MSE_exp[sl], marker=".", linestyle="None", markersize=markersize, color=colors_lst[0], 
                label="loss MSE")
    axs[1].plot(t[sl], loss_MSE_sim[sl], color=colors_lst[0])
    axs[1].plot(t[sl], np.zeros(len(t[sl])), color=colors_lst[0], linestyle="--")
    axs[1].set_xlabel("t", fontsize=font_size)
    axs[1].set_ylabel("Loss", fontsize=font_size)
    axs[1].set_ylim([-2, 2])
    axs[1].legend(loc="best")

    axs[-1].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    if save:
        plt.savefig("importants.png", dpi=300, bbox_inches="tight")
    plt.show()


# --------------------------------------------------------
# Animations
# --------------------------------------------------------
def animate_arm_w_arcs(traj_pos, L, Fx: Optional[NDArray] = None, Fy: Optional[NDArray] = None, frames=10,
                       interval_ms=30, save_path=None, fps=30, buckle_traj=None):
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
    colors_lst, red, _ = colors.color_scheme()
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", colors_lst)

    color_line = red
    color_arrow = 'k'
    tip_fontsize = 14

    pos = np.asarray(traj_pos, dtype=float)  # (T, N, 2)
    T_all = pos.shape[0]
    N_all = pos.shape[1]
    Edges = N_all - 1
    assert pos.ndim == 3 and pos.shape[2] == 2

    if np.shape(buckle_traj)[0] != np.shape(traj_pos)[0]:
        buckle_traj = np.tile(buckle_traj, np.shape(traj_pos)[0]).T.reshape(np.shape(traj_pos)[0],
                                                                            np.shape(traj_pos)[1]-2, 1)

    # --- downsample time ---
    stride = max(1, int(T_all / frames))
    pos = pos[::stride]
    max_F = 0.0
    if Fx is not None:
        Fx = Fx[::stride]
        max_Fx = max(abs(Fx))
        max_F = max_Fx
    if Fy is not None:
        Fy = Fy[::stride]
        max_Fy = max(abs(Fy))
        max_F = max([max_F, max_Fy])
    T, N, _ = pos.shape

    # ------ figure ------
    fig, (ax_chain, ax_force) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [1.1, 1.0]})

    # ------ left panel: chain ------
    ax_chain.set_aspect("equal", adjustable="box")
    ax_chain.set_xlim([-(Edges-0.5) * L, (Edges+0.5) * L])
    ax_chain.set_ylim([-(Edges-0.5) * L, (Edges-0.5) * L])
    ax_chain.set_xlabel("x")
    ax_chain.set_ylabel("y")

    # Polyline + joints + tip label
    (line,) = ax_chain.plot([], [], linewidth=4, color=color_line)
    scat = ax_chain.scatter([], [], s=60, zorder=3, color=color_line)
    tip_text = ax_chain.text(0, 0, "", va="bottom", ha="left", fontsize=tip_fontsize)

    # List to hold current arc patches so we can remove them each frame
    arc_patches: list[patches.Arc] = []

    # ------ right panel: force ------
    t_plot = np.arange(T)
    # start as bullets only
    (line_fx,) = ax_force.plot([], [], linestyle="-", marker="o", markersize=6, label=r"$F_x$")
    (line_fy,) = ax_force.plot([], [], linestyle="-", marker="o", markersize=6, label=r"$F_y$")
    # ax_force.set_ylim([-1.2*max_F, 1.2*max_F])
    ax_force.set_ylim([-600, 600])
    ax_force.set_xlim([-1, T+1])

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
        ax_chain.set_title(f"t= {ti + 1}/{T}")

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
                arrow = patches.FancyArrowPatch(p, p + v/np.linalg.norm(v)*0.004*N_all, arrowstyle='-|>',
                                                mutation_scale=25, linewidth=2, capstyle='round', joinstyle='round',
                                                color=color_arrow)
                try:
                    ax_chain.add_patch(arrow)
                    arc_patches.append(arrow)
                except:
                    print('bad animation, lets solve this later')

        if Fx is not None and Fy is not None:
            # ---- force history up to current frame ----
            tt = t_plot[: ti + 1]
            line_fx.set_data(tt, Fx[: ti + 1])
            line_fy.set_data(tt, Fy[: ti + 1])

        return line, scat, tip_text, *arc_patches

    ax_force.legend()

    anim = FuncAnimation(fig, update, frames=T, init_func=init, interval=interval_ms, blit=True)

    if save_path is not None:
        if save_path.lower().endswith(".gif"):
            anim.save(save_path, writer=PillowWriter(fps=fps))
        elif save_path.lower().endswith(".html"):
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(anim.to_jshtml())
        elif save_path.lower().endswith(".mp4"):  # doesn't work as of 2026Apr14
            anim.save(save_path, writer="ffmpeg", fps=fps)
        else:
            raise ValueError("save_path must end with .gif or .mp4")

    plt.close(fig)
    return fig, anim


def make_jpg_slider_html(
    frames_dir,
    html_path="animation_slider.html",
    frame_pattern="frame_{:04d}.jpg",
    n_frames=None,
    title="Training animation",
):
    frames_dir = Path(frames_dir)
    html_path = Path(html_path)

    if n_frames is None:
        frames = sorted(frames_dir.glob("*.jpg"))
        n_frames = len(frames)

    rel_frames_dir = frames_dir.relative_to(html_path.parent) if frames_dir.is_relative_to(html_path.parent) else frames_dir

    frame_list = [
        str(rel_frames_dir / frame_pattern.format(t)).replace("\\", "/")
        for t in range(n_frames)
    ]

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
    body {{
        font-family: Arial, sans-serif;
        margin: 20px;
    }}
    img {{
        max-width: 100%;
        border: 1px solid #ccc;
    }}
    .controls {{
        margin-top: 15px;
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    input[type=range] {{
        width: 500px;
    }}
    #tlabel {{
        font-size: 18px;
        font-weight: bold;
    }}
</style>
</head>
<body>

<h2>{title}</h2>

<img id="frame" src="{frame_list[0]}">

<div class="controls">
    <span>t =</span>
    <span id="tlabel">0</span>
    <input id="slider" type="range" min="0" max="{n_frames - 1}" value="0" step="1">
</div>

<script>
const frames = {frame_list};

const img = document.getElementById("frame");
const slider = document.getElementById("slider");
const tlabel = document.getElementById("tlabel");

slider.addEventListener("input", function() {{
    const t = Number(slider.value);
    img.src = frames[t];
    tlabel.textContent = t;
}});
</script>

</body>
</html>
"""

    html_path.write_text(html, encoding="utf-8")
    return html_path


def save_tip_update_jpg_frames(traj_pos, L, frames_dir: Union[str, Path], Fx: Optional[NDArray] = None,
                               Fy: Optional[NDArray] = None, frames: Optional[int] = None,
                               buckle_traj: Optional[NDArray] = None, dpi: int = 150,
                               jpeg_quality: int = 85, clear_existing: bool = True) -> list[Path]:
    """
    Save compressed JPEG frames for a tip-update trajectory.

    Parameters
    ----------
    traj_pos
        Array-like, shape ``(T, nodes, 2)``, arm positions over time.
    L
        Reference link length used for axis scaling.
    frames_dir
        Directory where ``frame_XXXX.jpg`` files are written. Created if needed.
    Fx, Fy
        Optional force histories aligned with ``traj_pos``.
    frames
        Approximate number of saved frames after temporal downsampling. If ``None``,
        save every input frame.
    buckle_traj
        Optional buckle history with shape ``(T, H, S)`` or static buckle state ``(H, S)``.
    dpi
        Output image resolution.
    jpeg_quality
        JPEG compression quality, from 1 to 95.
    clear_existing
        If True, remove existing ``frame_*.jpg`` files from ``frames_dir`` before saving.

    Returns
    -------
    list[Path]
        Paths to the saved JPEG files.
    """
    colors_lst, red, _ = colors.color_scheme()
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", colors_lst)

    pos = np.asarray(traj_pos, dtype=float)
    if pos.ndim != 3 or pos.shape[2] != 2:
        raise ValueError("traj_pos must have shape (T, nodes, 2)")

    T_all = pos.shape[0]
    if T_all == 0:
        raise ValueError("traj_pos must contain at least one frame")

    if frames is None:
        stride = 1
    else:
        stride = max(1, int(T_all / max(1, int(frames))))

    pos = pos[::stride]
    T, N, _ = pos.shape
    edges = N - 1

    Fx_plot = None if Fx is None else np.asarray(Fx, dtype=float)[::stride]
    Fy_plot = None if Fy is None else np.asarray(Fy, dtype=float)[::stride]

    buckle_plot = None
    if buckle_traj is not None:
        buckle_plot = np.asarray(buckle_traj)
        if buckle_plot.shape[0] != T_all:
            buckle_plot = np.tile(buckle_plot, T_all).T.reshape(T_all, N - 2, 1)
        buckle_plot = buckle_plot[::stride]

    frames_path = Path(frames_dir)
    frames_path.mkdir(parents=True, exist_ok=True)
    if clear_existing:
        for old_frame in frames_path.glob("frame_*.jpg"):
            old_frame.unlink()

    fig, (ax_chain, ax_force) = plt.subplots(1, 2, figsize=(10, 4.5),
                                             gridspec_kw={"width_ratios": [1.1, 1.0]})
    ax_chain.set_aspect("equal", adjustable="box")
    ax_chain.set_xlim([-(edges - 0.5) * L, (edges + 0.5) * L])
    ax_chain.set_ylim([-(edges - 0.5) * L, (edges - 0.5) * L])
    ax_chain.set_xlabel("x")
    ax_chain.set_ylabel("y")

    (chain_line,) = ax_chain.plot([], [], linewidth=4, color=red)
    chain_scat = ax_chain.scatter([], [], s=60, zorder=3, color=red)
    tip_text = ax_chain.text(0, 0, "", va="bottom", ha="left", fontsize=14)

    t_plot = np.arange(T)
    (line_fx,) = ax_force.plot([], [], linestyle="-", marker="o", markersize=6, label=r"$F_x$")
    (line_fy,) = ax_force.plot([], [], linestyle="-", marker="o", markersize=6, label=r"$F_y$")
    ax_force.set_ylim([-600, 600])
    ax_force.set_xlim([-1, T + 1])
    ax_force.legend()

    arc_patches: list[patches.Patch] = []
    saved_paths: list[Path] = []
    pad = max(4, len(str(T)))
    pil_kwargs = {"quality": int(np.clip(jpeg_quality, 1, 95)), "optimize": True}

    for ti in range(T):
        pts = pos[ti]
        xs, ys = pts[:, 0], pts[:, 1]
        chain_line.set_data(xs, ys)
        chain_scat.set_offsets(pts)
        tip_text.set_position((xs[-1], ys[-1]))
        tip_text.set_text(f"Tip ({xs[-1]:.2f}, {ys[-1]:.2f})")
        ax_chain.set_title(f"t= {ti + 1}/{T}")

        for patch in arc_patches:
            patch.remove()
        arc_patches.clear()

        if buckle_plot is not None:
            buckle = np.asarray(buckle_plot[ti])
            diffs = pts[2:, :] - pts[:-2, :]
            diffs_3d = np.concatenate((diffs, np.zeros((diffs.shape[0], 1))), axis=1)
            buckle_3d = np.concatenate((np.zeros((buckle.shape[0], 2)), buckle.reshape(-1, 1)), axis=1)
            directions = np.cross(diffs_3d, buckle_3d)[:, :2]
            for p, v in zip(pts[1:-1], directions):
                norm_v = np.linalg.norm(v)
                if norm_v < 1e-12:
                    continue
                arrow = patches.FancyArrowPatch(p, p + v / norm_v * 0.004 * N, arrowstyle="-|>",
                                                mutation_scale=25, linewidth=2, capstyle="round",
                                                joinstyle="round", color="k")
                ax_chain.add_patch(arrow)
                arc_patches.append(arrow)

        if Fx_plot is not None and Fy_plot is not None:
            tt = t_plot[:ti + 1]
            line_fx.set_data(tt, Fx_plot[:ti + 1])
            line_fy.set_data(tt, Fy_plot[:ti + 1])

        frame_path = frames_path / f"frame_{ti:0{pad}d}.jpg"
        fig.savefig(frame_path, dpi=dpi, bbox_inches="tight", format="jpg", pil_kwargs=pil_kwargs)
        saved_paths.append(frame_path)

    plt.close(fig)
    return saved_paths


# ----------------------------
# Post Processing
# ----------------------------
def plot_loss_columns(loss_columns: NDArray, Hamming_columns: Optional[NDArray] = None,
                      buckle_pairs: Optional[NDArray] = None, log_norm: bool = True,
                      save_path: Optional[str] = None, ax=None, ax2=None) -> None:
    """
    Plot initial/final MSE loss columns, optionally beside initial/final Hamming distance columns.

    Parameters
    ----------
    loss_columns    : ndarray, shape (N, 2)
        Initial and final MSE loss for each run.
    Hamming_columns : ndarray, shape (N, 2), optional
        Initial and final Hamming distance for each run.
    buckle_pairs    : ndarray, shape (N, 2), optional
        String labels ``[initial_buckle, desired_buckle]`` for every row.
    log_norm        : bool, default=True
        If True, use logarithmic color scaling for positive MSE losses.
    save_path       : str | None
        Optional path for saving the figure.
    ax, ax2         : matplotlib axes, optional
        Existing axes. ``ax2`` is used for Hamming distance.
    """
    loss_columns = np.asarray(loss_columns, dtype=float)
    if loss_columns.ndim != 2 or loss_columns.shape[1] != 2:
        raise ValueError("loss_columns must have shape (N, 2)")

    has_hamming = Hamming_columns is not None
    if has_hamming:
        Hamming_columns = np.asarray(Hamming_columns, dtype=float)
        if Hamming_columns.shape != loss_columns.shape:
            raise ValueError("Hamming_columns must have the same shape as loss_columns")

    _, _, custom_cmap = colors.color_scheme()

    created_fig = ax is None
    if created_fig:
        row_height = 0.18
        fig_height = max(3.0, min(14.0, row_height * loss_columns.shape[0]))
        if has_hamming:
            fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.5, fig_height), sharey=True,
                                          gridspec_kw={"width_ratios": [1.0, 1.0]})
        else:
            fig, ax = plt.subplots(figsize=(3.8, fig_height))
    else:
        fig = ax.figure
        if has_hamming and ax2 is None:
            ax2 = fig.add_subplot(1, 2, 2, sharey=ax)

    loss_plot_values = loss_columns.copy()
    norm = None
    if log_norm:
        positive = loss_plot_values[loss_plot_values > 0]
        if positive.size:
            floor = positive.min() * 0.5
            loss_plot_values[loss_plot_values <= 0] = floor
            norm = LogNorm(vmin=floor, vmax=loss_plot_values.max())

    im = ax.imshow(loss_plot_values, aspect="auto", cmap=custom_cmap, norm=norm)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Initial", "Final"])
    ax.set_title("MSE loss")

    if buckle_pairs is not None:
        buckle_pairs = np.asarray(buckle_pairs, dtype=str)
        ax.set_yticks(np.arange(buckle_pairs.shape[0]))
        ax.set_yticklabels([f"{init}->{desired}" for init, desired in buckle_pairs])
        ax.set_ylabel("initial -> desired buckle")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("loss_MSE")

    if has_hamming:
        finite_hamming = Hamming_columns[np.isfinite(Hamming_columns)]
        hamming_vmax = max(1.0, finite_hamming.max()) if finite_hamming.size else 1.0
        im2 = ax2.imshow(Hamming_columns, aspect="auto", cmap=custom_cmap, vmin=0, vmax=hamming_vmax)
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(["Initial", "Final"])
        ax2.set_title("Hamming distance")
        ax2.tick_params(axis="y", labelleft=False)

        cbar2 = fig.colorbar(im2, ax=ax2)
        cbar2.set_label("Hamming distance")

    fig.suptitle("Initial and final training metrics")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if created_fig:
        plt.show()


def plot_success_matrix(M: NDArray, N: int = 16, M_flag: Optional[NDArray] = None, M_flip: Optional[NDArray] = None) -> None:
    """
    Roie - document!
    """
    # ------ colors ------
    colors_lst, _, custom_cmap = colors.color_scheme()
    flag_color = colors_lst[4]
    flip_color = colors_lst[0]

    # ------ labels ------
    labels = []
    for i in range(N):
        b = format(i, "04b")
        labels.append(b)

    # ------ diagonal is white ------
    M_masked = np.ma.masked_where(np.eye(M.shape[0], dtype=bool), M)

    # ------ initialized fig ------
    plt.figure(figsize=(5, 5))

    # ------ plots ------
    # success matrix
    im = plt.imshow(M_masked, cmap=custom_cmap, vmin=0, vmax=4, origin="lower")

    # plot flagged runs (intersections)
    if M_flag is not None:
        nrows, ncols = M_flag.shape
        for i in range(nrows):
            for j in range(ncols):
                if M_flag[i, j]:
                    plt.plot([j - 0.35, j + 0.35], [i - 0.35, i + 0.35], color=flag_color, linestyle="-", linewidth=2.0)
                    # plt.plot([j - 0.35, j + 0.35], [i + 0.35, i - 0.35], color=flag_color, linestyle="-", linewidth=2.0)

    # plot runs that ended in chain symmetrical to desired
    if M_flip is not None:
        nrows, ncols = M_flip.shape
        for i in range(nrows):
            for j in range(ncols):
                if M_flip[i, j]:
                    plt.plot([j - 0.35, j + 0.35], [i + 0.35, i - 0.35], color=flip_color, linestyle="-", linewidth=2.0)
                    # plt.plot([j - 0.35, j + 0.35], [i + 0.35, i - 0.35], color=flip_color, linestyle="-", linewidth=2.0)

    # ------ ticks, legend and labels ------
    plt.xticks(range(N), labels, rotation=90)
    plt.yticks(range(N), labels)
    plt.xlabel("desired buckle")
    plt.ylabel("initial buckle")
    plt.title("Training success matrix")
    legend_elements = [patches.Patch(facecolor=custom_cmap(im.norm(0)), label="Success"),
                       patches.Patch(facecolor=custom_cmap(im.norm(1)), label="Missing"),
                       patches.Patch(facecolor=custom_cmap(im.norm(2)), label="Failure")]
    if M_flag is not None:
        legend_elements.append(Line2D([0], [0], color=flag_color, linestyle="-", linewidth=2, label="Self-intersection"))
    if M_flip is not None:
        legend_elements.append(Line2D([0], [0], color=flip_color, linestyle="-", linewidth=2, label="Symmetrical chain"))
    plt.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1))

    # ------ show ------
    plt.tight_layout()
    plt.show()


def plot_success_matrix_with_pathways(M_corr: np.ndarray, N: int, title: str = "Training success matrix (pathways corrected)"):
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

    labels = [helpers_builders.index_to_buckle(i) for i in range(N)]
    ax.set_xticks(np.arange(N))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(np.arange(N))
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


def plot_transition_diagram(transitions: Counter, *, transitions_between_runs: bool = True,
                            only_reached_nodes: bool = False, edge_zero_loss_count=None, missing_edges=None,
                            layout: str = "layers"):
    """
    Plot a directed buckle-transition diagram.

    Parameters
    ----------
    transitions : Counter
        Directed transition counts keyed by ``(src, dst)`` integer state pairs.
    transitions_between_runs : bool
        If True, color zero-loss initial-to-final transitions as successes.
    only_reached_nodes : bool
        If True, show only nodes touched by observed transitions.
    edge_zero_loss_count : Counter, optional
        Counts of zero-loss files per transition.
    missing_edges : iterable, optional
        Directed transitions to draw as dashed missing edges.
    layout : str
        ``"layers"``/``"hamming"`` arranges states by Hamming weight; ``"ring"`` arranges states on a ring.
    """
    colors_lst, _, _ = colors.color_scheme()
    occured_clr = colors_lst[0]
    success_clr = colors_lst[1]
    missing_clr = colors_lst[4]
    node_perim = "black"
    node_face = "white"
    text_color = "black"

    all_edges = list(transitions.keys()) + list(missing_edges or [])
    if not all_edges:
        raise ValueError("No transitions or missing_edges were provided, so the number of states cannot be inferred.")

    H = max(max(i, j) for (i, j) in all_edges).bit_length()
    H = max(H, 1)

    pos = helpers_builders.state_positions(H, layout=layout)

    if layout == "ring":
        fig, ax = plt.subplots(figsize=(10, 10))
        node_width = 0.11
        node_height = 0.08
        node_fontsize = 14 if H >= 5 else 18
        missing_alpha = 0.2
        missing_lw = 0.8
    else:
        fig, ax = plt.subplots(figsize=(12, 8))
        node_width = 0.1
        node_height = 0.08
        node_fontsize = 18
        missing_alpha = 0.75
        missing_lw = 1.5

    # ---- nodes ----
    if only_reached_nodes:
        used_nodes = set()
        for a, b in transitions:
            used_nodes.add(helpers_builders.index_to_buckle(a, H))
            used_nodes.add(helpers_builders.index_to_buckle(b, H))
    else:
        used_nodes = set(helpers_builders.all_binary_states(H))

    for s in helpers_builders.all_binary_states(H):
        if s not in used_nodes:
            continue
        x, y = pos[s]
        node = Ellipse((x, y), width=node_width, height=node_height, facecolor=node_face, edgecolor=node_perim, lw=2.5)
        ax.add_patch(node)
        ax.text(x, y, s, ha="center", va="center", fontsize=node_fontsize, color=text_color)

    # ---- edges ----
    if transitions:
        max_count = max(transitions.values())
    else:
        max_count = 1

    if edge_zero_loss_count is None:
        edge_zero_loss_count = Counter()

    for (src, dst), count in transitions.items():
        source = helpers_builders.index_to_buckle(src, H)
        dist = helpers_builders.index_to_buckle(dst, H)
        x1, y1 = pos[source]
        x2, y2 = pos[dist]

        rev_exists = (dst, src) in transitions
        rad = 0.18 if rev_exists and src < dst else (-0.18 if rev_exists else 0.0)

        lw = 2.5 + 4.0 * count / max_count

        if edge_zero_loss_count[(src, dst)] > 0:
            edge_color = success_clr   # or "cyan"
        else:
            edge_color = occured_clr

        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=22, lw=lw, color=edge_color,
                                shrinkA=20, shrinkB=20, connectionstyle=f"arc3,rad={rad}")
        ax.add_patch(arrow)

    if missing_edges:
        observed_edges = set(transitions.keys())

        for (src, dst) in missing_edges:
            source = helpers_builders.index_to_buckle(src, H)
            dist = helpers_builders.index_to_buckle(dst, H)
            x1, y1 = pos[source]
            x2, y2 = pos[dist]

            rev_exists = (dst, src) in observed_edges or (dst, src) in missing_edges
            rad = 0.12 if rev_exists and src < dst else (-0.12 if rev_exists else 0.0)

            arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=missing_lw,
                                    linestyle="--", color=missing_clr, alpha=missing_alpha, shrinkA=22, shrinkB=22,
                                    connectionstyle=f"arc3,rad={rad}", zorder=0)
            ax.add_patch(arrow)

    # ------- legend ------
    legend_handles = []

    if transitions_between_runs:
        legend_handles += [Line2D([0], [0], color=success_clr, lw=3, label="successful transition"),
                           Line2D([0], [0], color=occured_clr, lw=3, label="unintentional")]
    else:
        legend_handles += [Line2D([0], [0], color=occured_clr, lw=3, label="occurred")]

    legend_handles += [Line2D([0], [0], color=missing_clr, lw=2, linestyle="--", label="missing")]

    ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=14)

    if used_nodes:
        visible_xy = np.array([pos[s] for s in used_nodes], dtype=float)
    else:
        visible_xy = np.array(list(pos.values()), dtype=float)

    x_min, y_min = visible_xy.min(axis=0)
    x_max, y_max = visible_xy.max(axis=0)
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    span = max(x_max - x_min, y_max - y_min, node_width, node_height)
    margin = 0.25 * span if layout == "ring" else 0.12 * span
    half_span = 0.5 * span + margin

    ax.set_xlim(x_center - half_span, x_center + half_span)
    ax.set_ylim(y_center - half_span, y_center + half_span)
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
