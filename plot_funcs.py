from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from IPython.display import HTML
from matplotlib import patches
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation, PillowWriter  # for GIF export
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter
from matplotlib.patches import Ellipse, FancyArrowPatch
from collections import Counter
from pathlib import Path

from typing import Tuple, List, Union
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional, Sequence

import colors, helpers_builders

colors_lst, red, custom_cmap, shim = colors.color_scheme(add_shim=True)

# -------------------------------------------------
# Plot importants
# -------------------------------------------------
def plot_accuracy_afo_H(Hs: NDArray, accuracy: NDArray, bars: bool = False) -> tuple:
    """Plot accuracy as a function of ``H``, using bars or a dot-and-line curve."""
    Hs = np.asarray(Hs)
    accuracy = np.asarray(accuracy)
    if Hs.ndim != 1 or accuracy.ndim != 1 or Hs.shape != accuracy.shape:
        raise ValueError("Hs and accuracy must be one-dimensional arrays of equal length.")

    colors_lst, _, _ = colors.color_scheme()
    fig, ax = plt.subplots(figsize=(5, 2.5))
    if bars:
        ax.bar(Hs, accuracy, color=colors_lst[0], width=0.65)
    else:
        ax.plot(Hs, accuracy, color=colors_lst[0], marker="o", markersize=6, linewidth=2)
    ax.set_xlabel(r"$H$")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    plt.show()
    return fig, ax


def plot_accuracy_loss_hamming_summary(
        Hs: NDArray,
        accuracy: NDArray,
        loss_columns: Sequence[NDArray],
        hamming_columns: Sequence[NDArray],
        metric_Hs: Sequence[int],
        save_path: Optional[str | Path] = None,
        dpi: int = 300,
        font_size: float = 16.0
) -> tuple:
    """Plot accuracy and initial/final training metrics in one figure.

    The top panel shows accuracy as a function of hinge count. The bottom row
    follows :func:`plot_loss_columns`: MSE loss and Hamming distance each have
    initial and final columns, with one row per run. Loss normalization is
    shared across hinge counts, as is Hamming normalization.

    Parameters
    ----------
    Hs, accuracy
        One-dimensional arrays used in the accuracy panel.
    loss_columns, hamming_columns
        Matching sequences of ``(N, 2)`` initial/final metric arrays.
    metric_Hs
        Hinge count corresponding to each pair of metric arrays.
    save_path
        Optional output path.
    dpi
        Resolution used when saving the figure.

    Returns
    -------
    fig, axes
        The figure and a dictionary containing the top and bottom axes.
    """
    Hs = np.asarray(Hs)
    accuracy = np.asarray(accuracy, dtype=float)
    if Hs.ndim != 1 or accuracy.ndim != 1 or Hs.shape != accuracy.shape:
        raise ValueError("Hs and accuracy must be one-dimensional arrays of equal length.")
    if not (len(loss_columns) == len(hamming_columns) == len(metric_Hs)):
        raise ValueError("loss_columns, hamming_columns, and metric_Hs must have equal lengths.")
    if len(metric_Hs) == 0:
        raise ValueError("At least one bottom panel is required.")

    def validate_metric(metric: NDArray) -> NDArray:
        metric = np.asarray(metric, dtype=float)
        if metric.ndim != 2 or metric.shape[1] != 2:
            raise ValueError("Each metric array must have shape (N, 2).")
        return metric

    losses = []
    hammings = []
    for loss, hamming in zip(loss_columns, hamming_columns):
        loss = validate_metric(loss)
        hamming = validate_metric(hamming)
        if loss.shape != hamming.shape:
            raise ValueError("Each loss/Hamming pair must contain the same number of rows.")
        losses.append(loss.copy())
        hammings.append(hamming)

    positive_losses = np.concatenate([loss[np.isfinite(loss) & (loss > 0)] for loss in losses])
    if positive_losses.size:
        loss_floor = float(positive_losses.min()) * 0.5
        loss_norm = LogNorm(vmin=loss_floor, vmax=float(positive_losses.max()))
        for loss in losses:
            loss[loss <= 0] = loss_floor
    else:
        loss_norm = Normalize(vmin=0.0, vmax=1.0)

    finite_hamming = np.concatenate([hamming[np.isfinite(hamming)] for hamming in hammings])
    hamming_vmax = max(1.0, float(finite_hamming.max())) if finite_hamming.size else 1.0
    hamming_norm = Normalize(vmin=0.0, vmax=hamming_vmax)

    colors_lst, _, custom_cmap = colors.color_scheme()
    fig = plt.figure(figsize=(7.5, 10.0), constrained_layout=True)
    grid = fig.add_gridspec(
        4, 2 * len(metric_Hs), height_ratios=(1.0, 2.2, 0.10, 0.10))

    accuracy_ax = fig.add_subplot(grid[0, :])
    accuracy_ax.bar(Hs, accuracy, color=colors_lst[0], width=0.65)
    accuracy_ax.set(xlabel=r"$H$", ylabel="Accuracy", ylim=(0, 1))
    accuracy_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    accuracy_ax.spines[["top", "right"]].set_visible(False)
    accuracy_ax.tick_params(labelsize=font_size)
    accuracy_ax.xaxis.label.set_size(font_size)
    accuracy_ax.yaxis.label.set_size(font_size)

    loss_axes = []
    hamming_axes = []
    loss_image = None
    hamming_image = None
    for group, (hinges, loss, hamming) in enumerate(zip(metric_Hs, losses, hammings)):
        loss_ax = fig.add_subplot(grid[1, 2 * group])
        hamming_ax = fig.add_subplot(grid[1, 2 * group + 1], sharey=loss_ax)

        loss_image = loss_ax.imshow(
            loss, aspect="auto", cmap=custom_cmap, norm=loss_norm, interpolation="nearest")
        hamming_image = hamming_ax.imshow(
            hamming, aspect="auto", cmap=custom_cmap, norm=hamming_norm, interpolation="nearest")

        loss_ax.set_title(fr"$H={hinges}$ Loss", fontsize=font_size)
        hamming_ax.set_title(fr"$H={hinges}$ Hamming", fontsize=font_size)
        for ax in (loss_ax, hamming_ax):
            ax.set_xticks([0, 1], labels=["Initial", "Final"], fontsize=font_size)
            ax.tick_params(axis="y", labelsize=font_size)
        run_ticks = np.linspace(0, loss.shape[0] - 1, min(6, loss.shape[0]), dtype=int)
        if group == 0:
            loss_ax.set_ylabel("Run", fontsize=font_size)
            loss_ax.set_yticks(run_ticks, labels=(run_ticks + 1).astype(str))
        else:
            loss_ax.tick_params(axis="y", left=False, labelleft=False)
        hamming_ax.tick_params(axis="y", labelleft=False)

        loss_axes.append(loss_ax)
        hamming_axes.append(hamming_ax)

    loss_colorbar_ax = fig.add_subplot(grid[2, :])
    hamming_colorbar_ax = fig.add_subplot(grid[3, :])
    loss_colorbar = fig.colorbar(loss_image, cax=loss_colorbar_ax, orientation="horizontal")
    loss_colorbar.set_label("Loss", fontsize=font_size)
    loss_colorbar.ax.tick_params(labelsize=font_size)
    hamming_colorbar = fig.colorbar(
        hamming_image, cax=hamming_colorbar_ax, orientation="horizontal")
    hamming_colorbar.set_label("Hamming", fontsize=font_size)
    hamming_colorbar.ax.tick_params(labelsize=font_size)

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    return fig, {
        "accuracy": accuracy_ax,
        "loss": loss_axes,
        "hamming": hamming_axes,
        "colorbars": (loss_colorbar, hamming_colorbar),
    }


def plot_arm(
    pos_vec: np.ndarray,
    buckle: np.ndarray,
    L: float,
    modality: str,
    show: bool = True,
    ax=None,
    annotate_tip: bool = True,
    invert_x: bool = False,
    invert_y: bool = False,
    dpi: Optional[float] = None,
    x_lim: Optional[Sequence[float]] = None,
    y_lim: Optional[Sequence[float]] = None,
    font_size: Optional[float] = None,
    save: Optional[str] = None,
    show_encastre: bool = True,
) -> None:
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
    invert_x  - bool, mirror the plotted coordinates about the y-axis.
    invert_y  - bool, mirror the plotted coordinates about the x-axis.
    dpi       - Optional[float], figure resolution when creating new axes. For
                example, use ``dpi=300`` for a paper-ready raster figure.
    x_lim     - Optional two-value sequence with the fixed x-axis limits.
    y_lim     - Optional two-value sequence with the fixed y-axis limits.
    font_size - Optional font size for the title, axis labels, and tick labels.
    save      - Optional output format. Use ``"pdf"`` to save the figure in
                the current directory with a filename describing the chain.
    show_encastre - bool, draw a grey fixed support at the chain base. Its
                    diagonal hatching extends away from the chain.
    """
    # ------ prelims ------
    pos_vec = np.asarray(pos_vec).copy()
    N_nodes = pos_vec[:, 0].shape[0]

    if invert_x:
        pos_vec[:, 0] *= -1
    if invert_y:
        pos_vec[:, 1] *= -1

    # pick axes
    created_ax = ax is None
    if created_ax:
        _, ax = plt.subplots(figsize=(4, 4), dpi=dpi)

    xs, ys = pos_vec[:, 0], pos_vec[:, 1]
    tip_angle_deg = np.rad2deg(float(helpers_builders._get_tip_angle(pos_vec)))

    if modality in {"measurement", "desired"}:
        # clr = colors_lst[0]
        clr = red
    elif modality == "update":
        clr = colors_lst[2]
    else:
        clr = red

    # ------ fixed support at the chain base ------
    if show_encastre:
        support_color = "0.5"
        support_side = 1.0 if invert_x else -1.0
        support_half_height = 0.38 * L
        hatch_length = 0.16 * L
        hatch_rise = 0.12 * L

        ax.plot(
            [0.0, 0.0],
            [-support_half_height, support_half_height],
            color=support_color,
            linewidth=2.5,
            solid_capstyle="round",
            zorder=1,
        )
        for hatch_y in np.linspace(
                -support_half_height, support_half_height - hatch_rise, 5):
            ax.plot(
                [0.0, support_side * hatch_length],
                [hatch_y, hatch_y + hatch_rise],
                color=support_color,
                linewidth=1.5,
                solid_capstyle="round",
                zorder=1,
            )

    # ------ chain faces and nodes ------
    ax.plot(xs, ys, linewidth=4, color=clr)
    ax.scatter(xs, ys, s=60, zorder=3, color=clr)
    ax.scatter([0], [0], s=60, zorder=3, color="0.5")

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
                                        linewidth=3, capstyle="round", joinstyle="round", color=shim)
        try:
            ax.add_patch(arrow)
        except Exception:
            print("bad animation, lets solve this later")

    # ------ annotate tip and aesthetics -------
    if annotate_tip:
        ax.annotate(rf"$x={xs[-1]:.2f},\ y={ys[-1]:.2f},\ \theta={tip_angle_deg:.2f}$",
                    xy=(xs[-1], ys[-1]), xytext=(xs[-1] - 0.05, ys[-1] - 0.05))

    # aesthetics
    # ``adjustable="datalim"`` changes one of the requested limits at draw
    # time in order to obtain an equal aspect ratio.  Adjust the axes box
    # instead so explicitly supplied limits remain unchanged.
    ax.set_aspect("equal", adjustable="box")
    if x_lim is not None:
        ax.set_xlim(x_lim[0], x_lim[1])
    else:
        ax.set_xlim(xs.min() - 0.5 * L, xs.max() + 0.5 * L)
    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])
    else:
        ax.set_ylim(ys.min() - 0.5 * L, ys.max() + 0.5 * L)
    text_kwargs = {} if font_size is None else {"fontsize": font_size}
    ax.set_xlabel("x", **text_kwargs)
    ax.set_ylabel("y", **text_kwargs)
    ax.set_title(
        modality if modality is not None
        else None,
        **text_kwargs,
    )
    if font_size is not None:
        ax.tick_params(axis="both", labelsize=font_size)

    if save is not None:
        if save.lower() != "pdf":
            raise ValueError('plot_arm currently supports only save="pdf".')
        buckle_string = "_".join(
            str(int(value)) for value in np.asarray(buckle).reshape(-1)
        )
        inversion_label = "_inverted_x" if invert_x else ""
        filename = (
            f"chain_buckle={buckle_string}{inversion_label}"
            f"_x={xs[-1]:.3f}_y={ys[-1]:.3f}_theta={tip_angle_deg:.3f}.pdf"
        )
        ax.figure.savefig(filename, format="pdf", bbox_inches="tight")

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
                         save_path: Optional[str] = None, mod: Optional[str] = None,
                         tip_pos_des_in_t=None, tip_angle_des_in_t=None) -> None:
    """
    Plot the tip pose, forces, loss, tip updates, and buckle states over training.

    When ``mod="pos"``, label the top panel as the rest pose and omit the
    force panel.

    Parameters
    ----------
    loss_in_t   - np.ndarray, shape (T, 2)
    buckle_in_t - np.ndarray, shape (H, 1, T) or (T, H, 1)
    F_meas_in_t - np.ndarray, shape (T, 2)
    F_des_in_t  - np.ndarray, shape (T, 2)
    start       - int, inclusive
    end         - int, exclusive (None -> full length)
    mod         - str or None; ``"pos"`` selects the rest-position layout
    tip_pos_des_in_t - optional ndarray, shape (T, 2), desired tip positions
    tip_angle_des_in_t - optional ndarray, shape (T,), desired tip angles
    """
    # ------ colors ------
    colors_lst, _, _ = colors.color_scheme()
    zero_color = colors_lst[4]

    # -------- time vector / slicing and buckles --------
    T = np.shape(loss_in_t)[0]
    if end is None or end > T:
        end = T
    if start < 1:
        start = 1

    t = np.arange(start, end)

    buckle_in_t = _buckle_hinge_first(buckle_in_t, T)
    H = buckle_in_t.shape[0]

    # -------- instantiate plot --------
    position_mode = mod == "pos"
    n_panels = 4 if position_mode else 5
    fig, axes = plt.subplots(n_panels, 1, figsize=(6, 1.8 * n_panels), sharex=True)

    ax_pose = axes[0]
    if position_mode:
        ax_force = None
        ax_loss, ax_update, ax_buckle = axes[1:]
    else:
        ax_force, ax_loss, ax_update, ax_buckle = axes[1:]

    # -------- subplot 0: positions --------
    # ---- left axis: position ----
    position_labels = (r"$x^{rest}$", r"$y^{rest}$") if position_mode else (r"$tip_x$ meas", r"$tip_y$ meas")
    angle_label = r"$\theta^{rest}$" if position_mode else r"$\theta$ meas"
    ax_pose.plot(t, tip_pos_in_t[start:end, 0], color=colors_lst[1], label=position_labels[0])
    ax_pose.plot(t, tip_pos_in_t[start:end, 1], color=colors_lst[2], label=position_labels[1])
    if position_mode and tip_pos_des_in_t is not None:
        ax_pose.plot(t, tip_pos_des_in_t[start:end, 0], color=colors_lst[1], linestyle=':',
                     label=r"$x^{des}$")
        ax_pose.plot(t, tip_pos_des_in_t[start:end, 1], color=colors_lst[2], linestyle=':',
                     label=r"$y^{des}$")
    ax_pose.set_ylabel(r"$tip\left[mm\right]$")
    ax_pose.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ---- right axis: angle ----
    ax_pose_angle = ax_pose.twinx()
    ax_pose_angle.plot(t, tip_angle_in_t[start:end], color=colors_lst[3], label=angle_label)
    if position_mode and tip_angle_des_in_t is not None:
        ax_pose_angle.plot(t, tip_angle_des_in_t[start:end], color=colors_lst[3], linestyle=':',
                           label=r"$\theta^{des}$")
    ax_pose_angle.set_ylabel(r"$\theta\left[rad\right]$")

    # ---- combined legend ----
    lines = ax_pose.get_lines() + ax_pose_angle.get_lines()
    labels = [l.get_label() for l in lines]
    ax_pose.legend(lines, labels, ncol=3)

    # dashed at 0
    ax_pose.plot(t, np.zeros(end-start), color=zero_color, linestyle='--', alpha=0.6)

    # -------- subplot 1: forces --------
    if ax_force is not None:
        # measured (solid)
        ax_force.plot(t, F_meas_in_t[0, start:end], color=colors_lst[1], linestyle='-', label=r"$F_x$ meas")
        ax_force.plot(t, F_meas_in_t[1, start:end], color=colors_lst[2], linestyle='-', label=r"$F_y$ meas")

        # desired (dotted)
        ax_force.plot(t, F_des_in_t[0, start:end], color=colors_lst[1], linestyle=':', label=r"$F_x$ des")
        ax_force.plot(t, F_des_in_t[1, start:end], color=colors_lst[2], linestyle=':', label=r"$F_y$ des")
        ax_force.plot(t, np.zeros(end-start), color=zero_color, linestyle='--', alpha=0.6)

        ax_force.set_ylabel("Force [mN]")
        ax_force.legend(ncol=2)
        ax_force.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_force.set_ylim([-200, 500] if H < 6 else [-160, 160])

    # -------- subplot 2: loss --------
    loss_labels = [r'$L_x$', r'$L_y$', r'$L_\theta$']
    for i in range(min(loss_in_t.shape[1], len(loss_labels))):
        ax_loss.plot(t, loss_in_t[start:end, i], color=colors_lst[i + 1], label=loss_labels[i])
    loss_MSE_in_t = np.sqrt(np.sum(loss_in_t**2, axis=1))
    ax_loss.plot(t, loss_MSE_in_t[start:end], color=colors_lst[0], label=r'$\|L\|$')

    # dashed at 0
    ax_loss.plot(t, np.zeros(end-start), color=zero_color, linestyle='--', alpha=0.6)

    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_loss.set_ylim([-2.0, 4.0])

    # ------ subplot 3: delta tip update ------
    t_delta = t[1:]
    ax_update.plot(t_delta, np.diff(tip_pos_update_in_t[start:end, 0]), color=colors_lst[1],
                   label=r"$\Delta tip_x^{\,!}\left[mm\right]$")
    ax_update.plot(t_delta, np.diff(tip_pos_update_in_t[start:end, 1]), color=colors_lst[2],
                   label=r"$\Delta tip_y^{\,!}\left[mm\right]$")
    ax_update.set_ylabel(r"$\Delta tip^{\,!}\left[mm\right]$")
    ax_update.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ---- right axis: angle ----
    ax_update_angle = ax_update.twinx()
    ax_update_angle.plot(t_delta, np.diff(tip_angle_update_in_t[start:end]),
                         color=colors_lst[3], label=r"$\Delta\theta^{\,!}\left[rad\right]$")
    ax_update_angle.set_ylabel(r"$\Delta\theta^{\,!}\left[rad\right]$")

    lines = ax_update.get_lines() + ax_update_angle.get_lines()
    labels = [l.get_label() for l in lines]
    ax_update.legend(lines, labels, ncol=2)

    # dashed at 0
    ax_update.plot(t, np.zeros(end-start), color=zero_color, linestyle='--', alpha=0.6)

    # -------- subplot 4: buckle states --------
    for i in range(H):
        ax_buckle.plot(t, buckle_in_t[i, 0, start:end], color=colors_lst[i % len(colors_lst)],
                       label=f"hinge {i+1}")

    ax_buckle.set_ylabel("buckle")
    ax_buckle.set_xlabel("t")
    if H < 6:
        ax_buckle.legend()
    ax_buckle.xaxis.set_major_locator(MaxNLocator(integer=True))

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
                             save_path: Optional[str | Path] = None, show: bool = True,
                             font_size: Optional[float] = None, ax=None
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
    text_kwargs = {} if font_size is None else {"fontsize": font_size}
    ax.set_xlabel("tip angle [rad]", **text_kwargs)
    ax.set_ylabel("tip y [mm]", **text_kwargs)
    ax.set_title("Buckle state after update sweep", **text_kwargs)
    if font_size is not None:
        ax.tick_params(axis="both", labelsize=font_size)

    observed_ids = sorted(int(idx) for idx in np.unique(buckle_ids))
    handles = [
        patches.Patch(facecolor=cmap(norm(idx)), label=helpers_builders.index_to_buckle(idx, n_bits=n_bits))
        for idx in observed_ids
    ]
    if handles:
        legend_kwargs = {}
        if font_size is not None:
            legend_kwargs = {"fontsize": font_size, "title_fontsize": font_size}
        ax.legend(
            handles=handles,
            title="buckle",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            **legend_kwargs,
        )

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show and created_ax:
        plt.show()

    return fig, ax, buckle_ids


def plot_tip_grid_buckle_ids_from_npz(path_npz: str | Path, *,
                                      save_path: Optional[str | Path] = None,
                                      show: bool = True,
                                      font_size: Optional[float] = None, ax=None
                                      ) -> tuple[plt.Figure, plt.Axes, NDArray[np.int32]]:
    """Load and plot a completed tip-grid buckle-map archive.

    Archives written by ``file_funcs.export_tip_grid_buckle_map_npz`` are
    already arranged in canonical increasing-y/increasing-theta grid order.
    """
    with np.load(path_npz, allow_pickle=False) as data:
        buckle_matrix = np.asarray(data["buckle_matrix"], dtype=np.int32)
        grid_points = np.asarray(data["grid_points"], dtype=float)
        y_scale = float(data["y_scale"]) if "y_scale" in data else 1000.0

    if buckle_matrix.ndim != 4:
        raise ValueError(
            "Saved buckle_matrix must have shape (y_num, theta_num, hinges, shims)."
        )
    if grid_points.shape != (*buckle_matrix.shape[:2], 3):
        raise ValueError(
            "Saved grid_points must have shape (y_num, theta_num, 3) aligned "
            "with buckle_matrix."
        )

    y_num, theta_num = buckle_matrix.shape[:2]
    frames = buckle_matrix.reshape(y_num * theta_num, *buckle_matrix.shape[2:])
    return plot_tip_grid_buckle_ids(
        frames,
        y_num=y_num,
        theta_num=theta_num,
        theta_min=float(np.min(grid_points[:, :, 2])),
        theta_max=float(np.max(grid_points[:, :, 2])),
        y_min=float(np.min(grid_points[:, :, 1])),
        y_max=float(np.max(grid_points[:, :, 1])),
        grid_start=0,
        snake=False,
        y_scale=y_scale,
        save_path=save_path,
        show=show,
        font_size=font_size,
        ax=ax,
    )


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


def make_png_slider_html(frames_dir: str | Path, html_path: str | Path = "transition_animation.html",
                         glob_pattern: str = "*_init_*_desired_*.png",
                         title: str = "Transition animation") -> Path:
    """
    Create an HTML slider animation from saved PNG transition frames.

    Frames are sorted by the integer prefix before the first underscore, matching
    names such as ``0_init_0000_desired_1111.png``.
    """
    frames_dir = Path(frames_dir)
    html_path = Path(html_path)
    frames = sorted(
        frames_dir.glob(glob_pattern),
        key=lambda path: int(path.stem.split("_", 1)[0]),
    )
    if not frames:
        raise FileNotFoundError(f"No PNG frames matching {glob_pattern!r} in {frames_dir}")

    try:
        frame_list = [str(frame.relative_to(html_path.parent)).replace("\\", "/") for frame in frames]
    except ValueError:
        frame_list = [str(frame).replace("\\", "/") for frame in frames]

    labels = [frame.stem for frame in frames]
    n_frames = len(frames)

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
    <span>frame =</span>
    <span id="tlabel">0</span>
    <input id="slider" type="range" min="0" max="{n_frames - 1}" value="0" step="1">
</div>

<p id="caption">{labels[0]}</p>

<script>
const frames = {frame_list};
const labels = {labels};

const img = document.getElementById("frame");
const slider = document.getElementById("slider");
const tlabel = document.getElementById("tlabel");
const caption = document.getElementById("caption");

slider.addEventListener("input", function() {{
    const t = Number(slider.value);
    img.src = frames[t];
    tlabel.textContent = t;
    caption.textContent = labels[t];
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
                            layout: str = "layers", initial_state: int | str | None = None,
                            desired_state: int | str | None = None, title: str | None = None,
                            font_size: float = 18, ax=None,
                            show_legend: bool = True, show: bool = True):
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
    initial_state, desired_state : int | str, optional
        Buckle states to highlight. Initial node perimeter uses ``colors_lst[0]``;
        desired node perimeter uses ``colors_lst[1]``.
    title : str, optional
        Title displayed above the axes.
    font_size : float, default=18
        Font size for node labels, title, legend, and axis labels. Axis tick
        labels use ``font_size - 2``.
    ax : matplotlib.axes.Axes, optional
        Existing axes on which to draw. A new figure is created when omitted.
    show_legend : bool
        If True, draw the transition legend on these axes.
    show : bool
        If True, display the figure after drawing.
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

    def state_label(state) -> str | None:
        if state is None:
            return None
        if isinstance(state, str):
            return state
        return helpers_builders.index_to_buckle(int(state), H)

    initial_label = state_label(initial_state)
    desired_label = state_label(desired_state)

    pos = helpers_builders.state_positions(H, layout=layout)

    created_ax = ax is None
    if layout == "ring":
        if created_ax:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig = ax.figure
        node_width = 0.5
        node_height = 0.25
        node_fontsize = font_size
        missing_alpha = 0.8
        missing_lw = 1.2
    else:
        if created_ax:
            fig, ax = plt.subplots(figsize=(12, 8))
        else:
            fig = ax.figure
        node_width = 0.16
        node_height = 0.12
        node_fontsize = font_size
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
        edge_color = node_perim
        node_lw = 2.5
        if s == initial_label:
            edge_color = colors_lst[0]
            node_lw = 4.0
        if s == desired_label:
            edge_color = colors_lst[1]
            node_lw = 4.0
        node = Ellipse((x, y), width=node_width, height=node_height, facecolor=node_face, edgecolor=edge_color, lw=node_lw)
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

    if show_legend:
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            frameon=False,
            fontsize=font_size,
        )

    if used_nodes:
        visible_xy = np.array([pos[s] for s in used_nodes], dtype=float)
    else:
        visible_xy = np.array(list(pos.values()), dtype=float)

    x_min, y_min = visible_xy.min(axis=0)
    x_max, y_max = visible_xy.max(axis=0)
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    span = max(x_max - x_min, y_max - y_min, node_width, node_height)
    margin = 0.10 * span if layout == "ring" else 0.12 * span
    half_span = 0.5 * span + margin

    ax.set_xlim(x_center - half_span, x_center + half_span)
    ax.set_ylim(y_center - half_span, y_center + half_span)
    ax.set_aspect("equal")
    ax.tick_params(axis="both", labelsize=max(font_size - 2, 1))
    ax.axis("off")
    if title is not None:
        ax.set_title(title, fontsize=font_size, fontweight="bold", pad=12)
    if created_ax:
        fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_ring_transition_diagrams(
        position_transitions: Counter,
        force_transitions: Counter,
        sweep_transitions: Counter,
        *,
        edge_zero_loss_counts: Sequence[Counter | None] = (None, None, None),
        missing_edges: Sequence[Sequence[tuple[int, int]] | None] = (None, None, None),
        titles: Sequence[str] = ("Random", "Position", "Force", "Combined"),
        transitions_between_runs: bool = True,
        only_reached_nodes: bool = False,
        save_stem: str | Path | None = None,
        export_png: bool = False,
        export_eps: bool = False,
        export_pdf: bool = False,
        dpi: int = 300,
        font_size: float = 20,
        show: bool = True,
):
    """Plot random, position, force, and combined ring-transition diagrams.

    The combined panel merges position and force. An edge is intentional if
    either method has a zero-loss occurrence, and is missing only if both
    methods are missing it.

    Parameters
    ----------
    position_transitions, force_transitions, sweep_transitions : Counter
        Directed transition counts for the three panels.
    edge_zero_loss_counts : sequence of Counter or None
        Zero-loss transition counts, ordered as position, force, and sweep.
    missing_edges : sequence of edge sequences or None
        Missing directed edges for each panel, in the same order.
    titles : sequence of str
        Titles displayed above the four panels.
    transitions_between_runs, only_reached_nodes : bool
        Passed through to :func:`plot_transition_diagram`.
    save_stem : str or Path, optional
        Output path without an extension. A supplied suffix is replaced.
    export_png, export_eps, export_pdf : bool
        Export the figure in the selected format(s). ``save_stem`` is required
        when either option is enabled.
    dpi : int
        Resolution used for PNG export.
    font_size : float, default=20
        Shared font size for node labels, titles, and the figure legend.
    show : bool
        If True, display the completed figure.

    Returns
    -------
    tuple
        ``(fig, axes)`` for further notebook customization.
    """
    if len(edge_zero_loss_counts) != 3 or len(missing_edges) != 3:
        raise ValueError("edge_zero_loss_counts and missing_edges must each contain three items.")
    if len(titles) != 4:
        raise ValueError("titles must contain four items.")
    if (export_png or export_eps or export_pdf) and save_stem is None:
        raise ValueError("save_stem is required when an export format is enabled.")

    position_zero_loss = Counter(edge_zero_loss_counts[0] or {})
    force_zero_loss = Counter(edge_zero_loss_counts[1] or {})
    sweep_zero_loss = Counter(edge_zero_loss_counts[2] or {})
    combined_transitions = position_transitions + force_transitions
    combined_zero_loss = position_zero_loss + force_zero_loss
    combined_missing = sorted(
        set(missing_edges[0] or []) & set(missing_edges[1] or [])
    )

    transition_sets = (
        sweep_transitions,
        position_transitions,
        force_transitions,
        combined_transitions,
    )
    zero_loss_sets = (
        sweep_zero_loss,
        position_zero_loss,
        force_zero_loss,
        combined_zero_loss,
    )
    missing_sets = (
        missing_edges[2],
        missing_edges[0],
        missing_edges[1],
        combined_missing,
    )
    fig, axes = plt.subplots(1, 4, figsize=(26, 7))

    for ax_i, transitions_i, zero_loss_i, missing_i, title_i in zip(
            axes, transition_sets, zero_loss_sets, missing_sets, titles):
        plot_transition_diagram(
            transitions_i,
            transitions_between_runs=transitions_between_runs,
            only_reached_nodes=only_reached_nodes,
            edge_zero_loss_count=zero_loss_i,
            missing_edges=missing_i,
            layout="ring",
            title=title_i,
            font_size=font_size,
            ax=ax_i,
            show_legend=False,
            show=False,
        )

    colors_lst, _, _ = colors.color_scheme()
    legend_handles = [
        Line2D([0], [0], color=colors_lst[1], lw=4, label="successful transition"),
        Line2D([0], [0], color=colors_lst[0], lw=4,
               label="unintentional" if transitions_between_runs else "occurred"),
        Line2D([0], [0], color=colors_lst[4], lw=2, linestyle="--", label="missing"),
    ]
    if not transitions_between_runs:
        legend_handles.pop(0)
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles),
               frameon=False, fontsize=font_size, bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.91, bottom=0.15, wspace=-0.16)

    if save_stem is not None:
        output_stem = Path(save_stem).with_suffix("")
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        if export_png:
            fig.savefig(output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
        with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
            if export_eps:
                fig.savefig(output_stem.with_suffix(".eps"), format="eps", bbox_inches="tight", facecolor="white")
            if export_pdf:
                fig.savefig(output_stem.with_suffix(".pdf"), format="pdf", bbox_inches="tight", facecolor="white")

    if show:
        plt.show()
    return fig, axes


def plot_cumulative_transition_curve(coverage_df: pd.DataFrame, *,
                                     x_col: str = "training_task",
                                     y_col: str = "cumulative_unique_hamming_transitions",
                                     x_label: str | None = None,
                                     label: str | None = None,
                                     color_index: int = 0,
                                     save_path: str | Path | None = None,
                                     ax=None):
    """
    Plot cumulative transition coverage as a function of an ordered run step.

    This helper only draws the time/coverage curve. Building the cumulative
    table and transition diagrams stays in the calling notebook/script.
    Pass a shared ``ax`` with different labels and color indices to compare
    several coverage curves on one plot.
    """
    colors_lst, _, _ = colors.color_scheme()
    created_ax = ax is None
    if created_ax:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    ax.plot(
        coverage_df[x_col],
        coverage_df[y_col],
        color=colors_lst[color_index],
        marker="o",
        markerfacecolor=colors_lst[color_index],
        markeredgecolor=colors_lst[color_index],
        markersize=5,
        lw=2.5,
        label=label,
    )
    ax.set_xlabel(x_label if x_label is not None else x_col.replace("_", " "))
    ax.set_ylabel("cumulative Hamming transitions")
    ax.set_ylim(-2, 64)
    if not any(line.get_gid() == "transition-coverage-limit" for line in ax.lines):
        limit_line = ax.axhline(
            64, color=colors_lst[0], linestyle="--", linewidth=2, alpha=0.8
        )
        limit_line.set_gid("transition-coverage-limit")
    if label is not None:
        ax.legend(frameon=False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if created_ax:
        fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if created_ax:
        plt.show()
    return fig, ax


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
