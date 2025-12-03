from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp
from pathlib import Path
from scipy.signal import savgol_filter

from typing import Tuple, List
from numpy import array, zeros
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional

if TYPE_CHECKING:
    from SupervisorClass import SupervisorClass
    from VariablesClass import VariablesClass
    from EquilibriumClass import EquilibriumClass


# ===================================================
# file_funcs - functions to assist with file conversions etc.
# ===================================================


def export_stress_strain_sim(Sprvsr: "SupervisorClass", Fx_afo_pos: NDArray[np.float_], L: float, buckle_arr: NDArray[np.int],
                             filename: str = None) -> None:

    # --- build pandas dataframe ---
    df = pd.DataFrame({
        "x_tip": Sprvsr.tip_pos_in_t[:, 0],
        "y_tip": Sprvsr.tip_pos_in_t[:, 1],
        "tip_angle_rad": Sprvsr.tip_angle_in_t,
        "Fx": Fx_afo_pos,
    })
    if filename is not None:
        pass 
    else:
        filename = f"L={L}_buckle{buckle_arr.reshape(-1)}.csv"  # filename example "L=1_buckle1111.csv"
    out_path = Path(filename)
    df.to_csv(out_path, index=False)


def import_stress_strain_sim_and_plot(path: str, plot: bool = False) -> df:
    sim_df = pd.read_csv(path)   # assumes the header row is in the file
    if plot:
        plt.plot(sim_df['x_tip'], sim_df['Fx'])
        plt.xlabel('tip pos')
        plt.ylabel('Fx')
        plt.show()
    return sim_df


def import_stress_strain_exp_and_plot(path: str, plot: bool = True) -> None:
    exp_df = pd.read_csv(path)   # assumes the header row is in the file
    if plot:
        plt.plot(exp_df['Position (mm)'], exp_df['Load2 (N)'])
        plt.xlabel('tip pos')
        plt.ylabel('Fx')
        plt.show()
    return exp_df


def build_torque_and_k_from_file(path: str, *, contact: bool = True, angles_in_degrees: bool = True, savgol_window: int = None,
                                 contact_scale: float = 1e2) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray,
                                                                      callable, callable]:
    """
    Load (angle, torque) from text file and construct JAX-friendly interpolants.
    contact: bool = if True, account for contact between plastic faces where torques explode

    Returns
    -------
    theta_grid : jnp.ndarray, shape (N,)
        Sorted grid of angles (radians).
    torque_grid : jnp.ndarray, shape (N,)
        Torque samples on grid.
    k_grid : jnp.ndarray, shape (N,)
        Stiffness samples (numeric derivative of torque w.r.t. theta).
    torque_of_theta : callable
        JAX function theta -> torque (linear interpolation, clamped at ends).
    k_of_theta : callable
        JAX function theta -> stiffness (linear interpolation, clamped at ends).
    """
    # --- load ---
    data = np.loadtxt(path)                # shape (N, 2)
    theta = data[:, 0]
    tau = data[:, 1]

    # degrees -> radians if needed
    if angles_in_degrees:
        theta = np.deg2rad(theta)

    # --- sort & unique (interp requires monotonic x) ---
    order = np.argsort(theta)
    theta = theta[order]
    tau = tau[order]
    # collapse duplicates (if any)
    theta_u, idx = np.unique(theta, return_index=True)
    tau_u = tau[idx]

    # --- numeric derivative: k = d(tau)/d(theta) ---
    # central differences on nonuniform grid
    # (np.gradient handles non-uniform spacing if you pass x)
    k = np.gradient(tau_u, theta_u)

    # optional light smoothing of k (pure NumPy, outside JAX)
    if savgol_window is not None and savgol_window > 2 and savgol_window % 2 == 1:
        try:
            k = savgol_filter(k, window_length=savgol_window, polyorder=4, mode="interp")
        except Exception:
            print('SciPy isnt available, just skip smoothing')

    # --- wrap as JAX arrays ---
    theta_grid = jnp.asarray(theta_u, dtype=jnp.float32)
    torque_grid = jnp.asarray(tau_u, dtype=jnp.float32)
    k_grid = jnp.asarray(k, dtype=jnp.float32)
    k_grid = k_grid.at[k_grid < 0].set(10e-4)  # for numerical stability, singular point of experimental negative k

    # --- clamped linear interpolators (JAX) ---
    def _clamp(x, xmin, xmax):
        return jnp.clip(x, xmin, xmax)

    def torque_of_theta(theta_query: jnp.ndarray) -> jnp.ndarray:
        # masks for outside vs inside range
        above = theta_query > theta_grid[-1]
        below = theta_query < theta_grid[0]
        th = _clamp(theta_query, theta_grid[0], theta_grid[-1])
        tau = jnp.interp(th, theta_grid, torque_grid)  # torque
        if contact:  # account for plates in contact, torque diverges
            # masks for outside vs inside range
            above = theta_query > theta_grid[-1]
            below = theta_query < theta_grid[0]

            tau = jnp.where(above, contact_scale * jnp.max(torque_grid), tau)
            tau = jnp.where(below, contact_scale * jnp.min(torque_grid), tau)
        return tau

    def k_of_theta(theta_query: jnp.ndarray) -> jnp.ndarray:
        th = _clamp(theta_query, theta_grid[0], theta_grid[-1])
        return jnp.interp(th, theta_grid, k_grid)

    return theta_grid, torque_grid, k_grid, torque_of_theta, k_of_theta
