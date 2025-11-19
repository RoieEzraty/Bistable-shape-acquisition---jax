import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import patches
from matplotlib.animation import FuncAnimation, PillowWriter  # for GIF export
from scipy.signal import savgol_filter

from typing import List

import colors, helpers_builders


from typing import Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

import colors
import helpers_builders


def plot_arm(pos_vec: np.ndarray, buckle: np.ndarray, thetas: Union[np.ndarray, list, tuple], L: float, modality: str,
             arc_scale: float = 0.2) -> None:
    """
    Plot bistable hinge chain (arm) given all joint positions and hinge buckling states.

    Parameters
    ----------
    pos_vec : 2D np.ndarray, (N, 2), node coordinates
    buckle : 1D np.ndarray, (N-1,), with hinge buckling orientation, +1 is shim going down
    thetas : array-like of float, (N-1,), Hinge angles in radians
    L : float, edge length, used to scale the arcs and plot limits.
    modality : str, Plotting mode that sets the color scheme. Recognized values:
                    - "measurement" : primary color (e.g., blue)
                    - "update"      : secondary color (e.g., orange)
    arc_scale : float, optional, Relative radius of the plotted hinge arcs in units of L. Default is 0.2.

    Notes
    -----
    - The tip angle shown in the title is computed from the positions via
      `helpers_builders._get_tip_angle` (in radians) and converted to degrees
      only for display.
    - Hinge arcs are drawn such that the sign of `buckle[i]` controls whether
      the arc is visually clockwise or counter-clockwise: a value of -1 will
      swap the order of the start/end angles relative to +1.
    """
    colors_lst, red, custom_cmap = colors.color_scheme()
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", colors_lst)

    # Convert inputs to NumPy arrays for plotting
    pos = np.asarray(pos_vec, dtype=float)
    buckle_arr = np.asarray(buckle)
    thetas_rad = np.asarray(thetas, dtype=float)  # radians

    # Extract x, y and tip angle (converted to degrees only for display)
    xs, ys = pos[:, 0], pos[:, 1]
    tip_angle_deg = np.rad2deg(float(helpers_builders._get_tip_angle(pos_vec)))

    # ---- figure ----
    plt.figure(figsize=(4, 4))

    if modality == "measurement":
        clr = colors_lst[0]
    elif modality == "update":
        clr = colors_lst[2]
    else:
        # Fallback color if modality is unknown
        clr = colors_lst[1]

    # Plot polyline (all links)
    plt.plot(xs, ys, linewidth=4, color=clr)
    # Scatter joints
    plt.scatter(xs, ys, s=60, zorder=3, color=clr)
    # Origin in black
    plt.scatter([0], [0], s=60, zorder=3, color="k")

    # --- line of wall ---
    plt.plot([xs[-1], xs[-1]],  # vertical line at tip x
             [ys[-1] + 0.4 * L, ys[-1] - 0.4 * L],  # short segment
             linestyle=":", color="k", linewidth=3.0)

    # ---- draw hinge arcs with buckle-directed orientation ----
    r = arc_scale * float(L)

    # cumulative hinge angles in radians
    cumsum_thetas = np.cumsum(thetas_rad)
    # a second set shifted by pi (180 degrees) but still in radians
    cumsum_thetas_shift = cumsum_thetas - cumsum_thetas[0] + np.pi

    ax = plt.gca()
    for i in range(buckle_arr.size):
        p = pos[i + 1]

        if buckle_arr[i] == -1:
            theta1 = cumsum_thetas[i]
            theta2 = cumsum_thetas_shift[i]
        else:
            theta1 = cumsum_thetas_shift[i]
            theta2 = cumsum_thetas[i]

    # Convert to degrees only at the last moment for Matplotlib
        theta1_deg = float(np.rad2deg(theta1))
        theta2_deg = float(np.rad2deg(theta2))

        arc = patches.Arc(
            xy=(p[0], p[1]),
            width=2 * r,
            height=2 * r,
            angle=0.0,
            theta1=theta1_deg,
            theta2=theta2_deg,
            linewidth=2,
            zorder=2,
        )
        ax.add_patch(arc)
        ax.add_patch(arc)

    # annotate tip
    plt.annotate("Tip", xy=(xs[-1], ys[-1]), xytext=(xs[-1] + 0.05, ys[-1] + 0.05))

    # aesthetics
    plt.axis("equal")
    plt.xlim(xs.min() - 0.5 * L, xs.max() + 0.5 * L)
    plt.ylim(ys.min() - 0.5 * L, ys.max() + 0.5 * L)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Tip (x, y, theta)=({xs[-1]:.2f}, {ys[-1]:.2f}, {tip_angle_deg:.2f})")
    plt.show()


def animate_arm(traj_pos, L, frames=10, interval_ms=30, save_path=None, fps=30, show_inline=False):
    """
    Animate an N-link arm over time.
    traj_pos: array-like, shape (T, N, 2), positions over time
    L: reference link length (used only for nice padding if needed)
    interval_ms: delay between frames (for interactive playback)
    save_path: if provided, writes an animation ('.gif' or '.mp4')
    fps: frames per second when saving

    Returns: (fig, anim) so you can display or save later.
    """
    pos = np.asarray(traj_pos)              # (T, N, 2)
    T = np.shape(pos)[0]
    assert pos.ndim == 3 and pos.shape[2] == 2

    # --- downsample time ---
    stride = int(T/frames)
    pos = pos[::max(1, int(stride))]
    T, N, _ = pos.shape

    # Precompute axes limits from the entire trajectory (stable view)
    # x_min, x_max = pos[...,0].min(), pos[...,0].max()
    # y_min, y_max = pos[...,1].min(), pos[...,1].max()
    # pad = 0.25 * max(1e-6, x_max - x_min, y_max - y_min, L)
    # x_min, x_max = x_min - pad, x_max + pad
    # y_min, y_max = y_min - pad, y_max + pad
    xs, ys = pos[:, 0], pos[:, 1]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect('equal', adjustable='box')
    # ax.set_xlim(x_min, x_max)
    # ax.set_ylim(y_min, y_max)
    ax.set_xlim([-L, 8*L])
    ax.set_ylim([-4.5*L, 4.5*L])
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Polyline + joints + tip label
    line,   = ax.plot([], [], linewidth=4)
    scat     = ax.scatter([], [], s=60, zorder=3)
    tip_text = ax.text(0, 0, "", va="bottom", ha="left")

    def init():
        line.set_data([], [])
        scat.set_offsets(np.empty((0, 2)))
        tip_text.set_text("")
        return line, scat, tip_text

    def update(ti):
        pts = pos[ti]                    # (N, 2)
        xs, ys = pts[:, 0], pts[:, 1]
        line.set_data(xs, ys)
        scat.set_offsets(pts)
        tip_text.set_position((xs[-1], ys[-1]))
        tip_text.set_text(f"Tip ({xs[-1]:.2f}, {ys[-1]:.2f})")
        ax.set_title(f"Frame {ti+1}/{T}")
        return line, scat, tip_text

    anim = FuncAnimation(fig, update, frames=T, init_func=init,
                         interval=interval_ms, blit=True)

    if save_path is not None:
        if save_path.lower().endswith(".gif"):
            anim.save(save_path, writer=PillowWriter(fps=fps))
        elif save_path.lower().endswith(".mp4"):
            # Requires ffmpeg installed
            anim.save(save_path, writer="ffmpeg", fps=fps)
        else:
            raise ValueError("save_path must end with .gif or .mp4")

    # ---- Inline display (keep small!) ----
    if show_inline:
        # reduce embed size by downsampling and smaller fig/dpi
        from IPython.display import HTML
        return HTML(anim.to_jshtml())

    plt.close(fig)
    return fig, anim


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


# def plot_energies(Variabs: "VariablesClass", Strctr: "StructureClass", pos_in_t: np.array[np.float_], Energy_func, ):
#     T = np.shape(pos_in_t)[0]
#     energies = np.zeros(int(T))
#     for i in range(int(T)):
#         energies[i], _, _ = Energy_func(Variabs, Strctr, pos_in_t[i])
        
#     plt.plot(energies)
#     plt.yscale('log')
