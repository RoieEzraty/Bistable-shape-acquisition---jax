from __future__ import annotations

import time
import diffrax
import jax
import numpy as np
import jax.numpy as jnp
import equinox as eqx
from jax import grad, jit, vmap
from jax.experimental.ode import odeint

from typing import Tuple, List
from numpy import array, zeros
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional

if TYPE_CHECKING:
    from StructureClass import StructureClass
    from VariablesClass import VariablesClass
    from EqClass import EqClass


# ===================================================
# file_funcs - functions to assist with file conversions etc.
# ===================================================


def build_torque_stiffness_from_file(
    path: str,
    *,
    angles_in_degrees: bool = True,
    savgol_window: int = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, callable, callable]:
    """
    Load (angle, torque) from text file and construct JAX-friendly interpolants.

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
    tau   = data[:, 1]

    # degrees -> radians if needed
    if angles_in_degrees:
        theta = np.deg2rad(theta)

    # --- sort & unique (interp requires monotonic x) ---
    order = np.argsort(theta)
    theta = theta[order]
    tau   = tau[order]
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
            from scipy.signal import savgol_filter
            k = savgol_filter(k, window_length=savgol_window, polyorder=4, mode="interp")
            print('smoothed your k')
        except Exception:
            print('SciPy isnt available, just skip smoothing')

    # --- wrap as JAX arrays ---
    theta_grid  = jnp.asarray(theta_u, dtype=jnp.float32)
    torque_grid = jnp.asarray(tau_u,    dtype=jnp.float32)
    k_grid      = jnp.asarray(k,        dtype=jnp.float32)

    # --- clamped linear interpolators (JAX) ---
    def _clamp(x, xmin, xmax):
        return jnp.clip(x, xmin, xmax)

    def torque_of_theta(theta_query: jnp.ndarray) -> jnp.ndarray:
        th = _clamp(theta_query, theta_grid[0], theta_grid[-1])
        return jnp.interp(th, theta_grid, torque_grid)

    def k_of_theta(theta_query: jnp.ndarray) -> jnp.ndarray:
        th = _clamp(theta_query, theta_grid[0], theta_grid[-1])
        return jnp.interp(th, theta_grid, k_grid)

    return theta_grid, torque_grid, k_grid, torque_of_theta, k_of_theta
