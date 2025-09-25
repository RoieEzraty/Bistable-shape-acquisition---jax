import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import patches
from matplotlib.animation import FuncAnimation, PillowWriter  # for GIF export

import colors


def plot_arm(pos_vec: np.ndarray, buckle: np.array, thetas, L: float,  modality: str, arc_scale: float = 0.2) -> None:
    """
    Plot an N-link arm given all joint positions.
    
    pos_vec: (N,2) array of coordinates (JAX or NumPy).
             Row 0 is base, row N-1 is tip.
    L: reference length (used for scaling the arcs and axes).
    """

    colors_lst, red, custom_cmap = colors.color_scheme()
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', colors_lst)

    # convert to NumPy for plotting
    pos = np.asarray(pos_vec)

    # Extract x,y
    xs, ys = pos[:, 0], pos[:, 1]

    # ---- figure ----
    plt.figure(figsize=(4, 4))

    if modality == "measurement":
        clr = colors_lst[0]
    elif modality == "update":
        clr = colors_lst[2]
    # plot polyline (all links)
    plt.plot(xs, ys, linewidth=4, color=clr)
    # scatter joints
    plt.scatter(xs, ys, s=60, zorder=3, color=clr)
    # origin in black
    plt.scatter([0], [0], s=60, zorder=3, color='k')

    # --- line of wall --- 

    plt.plot(
    [xs[-1], xs[-1]],              # vertical line at tip x
    [ys[-1] + 0.4*L, ys[-1] - 0.4*L],      # short downward segment
    linestyle=":", color="k", linewidth=3.0
)

    # ---- draw hinge arcs with buckle-directed orientation ----
    r = arc_scale * float(L)

    cumsum_thetas1 = np.cumsum(thetas)
    # cumsum_thetas2 = cumsum_thetas1-cumsum_thetas1[0]+180
    cumsum_thetas2 = cumsum_thetas1-cumsum_thetas1[0]+180

    for i in range(0, buckle.size):
        p = pos[i+1]

        # if CW desired (sgn=-1), theta2 < theta1 makes the arc go CW visually
        if buckle[i] == -1:
            theta1 = cumsum_thetas1[i]
            theta2 = cumsum_thetas2[i]
        else:
            theta1 = cumsum_thetas2[i]
            theta2 = cumsum_thetas1[i]

        arc = patches.Arc(
            xy=(p[0], p[1]),
            width=2*r, height=2*r,
            angle=0.0,
            theta1=theta1, theta2=theta2,
            linewidth=2, zorder=2
        )
        plt.gca().add_patch(arc)

    # annotate tip
    plt.annotate("Tip", xy=(xs[-1], ys[-1]),
                 xytext=(xs[-1]+0.05, ys[-1]+0.05))

    # aesthetics
    plt.axis('equal')
    plt.xlim(xs.min() - 0.5*L, xs.max() + 0.5*L)
    plt.ylim(ys.min() - 0.5*L, ys.max() + 0.5*L)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Tip (x, y)=({xs[-1]:.2f}, {ys[-1]:.2f})")
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
    ax.set_xlim([-1, 8])
    ax.set_ylim([-4.5, 4.5])
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

    return fig, anim
