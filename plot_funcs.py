import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import colors

def plot_arm(pos_vec: jnp.ndarray, L: float) -> None:
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

    # plot polyline (all links)
    plt.plot(xs, ys, linewidth=4)

    # scatter joints
    plt.scatter(xs, ys, s=60, zorder=3, color=colors_lst[0])
    plt.scatter([0], [0], s=60, zorder=3, color='k')

    # annotate tip
    plt.annotate("Tip", xy=(xs[-1], ys[-1]),
                 xytext=(xs[-1]+0.05, ys[-1]+0.05))

    # aesthetics
    plt.axis('equal')
    plt.xlim(xs.min() - 0.5*L, xs.max() + 0.5*L)
    # plt.ylim(ys.min() - 0.5*L, ys.max() + 0.5*L)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Tip (x, y)=({xs[-1]:.2f}, {ys[-1]:.2f})")
    plt.show()
