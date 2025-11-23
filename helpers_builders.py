from __future__ import annotations

import numpy as np
import copy
import diffrax
import jax
import jax.numpy as jnp
import equinox as eqx
from jax import grad, jit, vmap

from typing import Tuple, List
from numpy import array, zeros
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional

import dynamics

if TYPE_CHECKING:
    from StructureClass import StructureClass
    from EquilibriumClass import EquilibriumClass
    from VariablesClass import VariablesClass


# --- convenience: move from jax to numpy arrays and vice-verse ---
def numpify(arr: jax.Array) -> NDArray[np.float_]:
    """
    Convert JAX array to NumPy array.

    Breaks JAX's tracing, so should only be used at the boundaries of your pipeline.
    """
    # return np.asarray(arr)
    return np.asarray(jax.device_get(arr))


def jaxify(arr: NDArray[np.float_]) -> jnp.ndarray:
    """
    Convert a NumPy array (or Python list) back to a JAX array.

    This creates a regular JAX array on the default device.
    """
    return jnp.asarray(arr)


# --- reshapes ---
def _reshape_pos_arr_2_state(pos_arr: jnp.Array[jnp.float_]) -> jnp.Array[jnp.float_]:
    """
    Flatten position array into a full state vector ([x0, y0, x1, y1, .... x0_dot, y0_dot...]).

    Parameters
    ----------
    pos_arr : (nodes, 2) jnp.ndarray
        Node positions, with each row containing (x, y) coordinates.

    Returns
    -------
    state : (2*N,) jnp.ndarray
        Concatenated state vector, where the first N*2 entries are the
        flattened positions and the last N*2 entries are initialized
        velocities (all zeros).
    """
    first_half = pos_arr.flatten()
    second_half = jnp.zeros_like(first_half)
    return jnp.concatenate([first_half, second_half])


def _reshape_state_2_pos_arr(state: jnp.Array[jnp.float_], pos_arr) -> jnp.Array[jnp.float_]:
    """
    Reshape a flattened state vector back into node positions.

    Parameters
    ----------
    state : (2*2*N,) jnp.ndarray
        Flattened state vector of positions and velocities.
    pos_arr : (N, 2) jnp.ndarray
        Template array defining the desired shape for positions.

    Returns
    -------
    pos_arr : (N, 2) jnp.ndarray
        Reshaped positions, extracted from the first N*2 entries of `state`.
    """
    return state.reshape(pos_arr.shape)


# --- initiate ---
def _initiate_pos(nodes: int, L: float, numpify: bool = False) -> jax.Array:
    """
    `(hinges+2, 2)` each pair is (xi, yi) of point i going like [[0, 0], [1, 0], [2, 0], etc]
    
    Parameters
    ----------
    hinges : int
        Number of hinges (internal joints). The number of nodes will be
        hinges + 2 (two end nodes + internal ones).

    Returns
    -------
    pos_arr : (hinges+2, 2) jnp.ndarray
        Node coordinates laid out on the x-axis, starting from (0,0).
        Example for hinges=2:
        [[0,0], [1,0], [2,0], [3,0]]
    """
    flat = L*jnp.arange(nodes, dtype=jnp.float32)
    pos_arr = jnp.stack([flat, jnp.zeros_like(flat)], axis=1)
    if numpify:
        return numpify(pos_arr)
    else:
        return pos_arr


def _initiate_buckle(hinges: int, shims: int, numpify: bool = False) -> jax.Array:
    """
    `(hinges+2, 2)` each pair is (xi, yi) of point i going like [[0, 0], [1, 0], [2, 0], etc]
    
    Parameters
    ----------
    hinges : int
        Number of hinges (internal joints). The number of nodes will be
        hinges + 2 (two end nodes + internal ones).

    Returns
    -------
    pos_arr : (hinges+2, 2) jnp.ndarray
        Node coordinates laid out on the x-axis, starting from (0,0).
        Example for hinges=2:
        [[0,0], [1,0], [2,0], [3,0]]
    """
    buckle = jnp.ones((hinges, shims))
    # pos_arr_in_t = copy.copy(pos_arr)
    # return pos_arr, pos_arr_in_t
    if numpify:
        return numpify(buckle)
    else:
        return buckle


# --- DOFs - free and essential ---
def _assemble_full(free_mask: jax.Array,       # bool (n_coords,)
                   fixed_mask: jax.Array,      # bool (n_coords,)
                   imposed_mask: jax.Array,    # bool (n_coords,)
                   x_free: jax.Array,
                   fixed_vals_t: jax.Array,
                   imposed_vals_t: jax.Array,  # (n_coords,)
                   ) -> jax.Array:
    """
    Build full flattened x from free DOFs + constraints at time t.

    Parameters
    ----------
    x_free : (nodes_free*2,) jnp.ndarray
        Values of the free degrees of freedom.
    free_mask : (nodes*2,) jnp.ndarray[bool]
            Boolean mask marking which entries are free DOFs.
    fixed_mask : (nodes*2,) jnp.ndarray[bool]
        Boolean mask marking fixed DOFs (always zero).
    imposed_mask : (nodes*2,) jnp.ndarray[bool]
        Boolean mask marking imposed DOFs (prescribed values).
    imposed_vals_t : (nodes*2,) jnp.ndarray
        Values to assign to imposed DOFs at time t.
    n_coords : int
        Total number of coordinates (flattened x and y for all nodes).

    Returns
    -------
    x_full : (n_coords,) jnp.ndarray
        Complete flattened coordinate vector, consistent with all masks
        and imposed values.
    """
    # x_full = jnp.zeros((n_coords,), dtype=imposed_vals_t.dtype)
    # x_full = x_full.at[free_mask].set(x_free)
    # x_full = jnp.where(fixed_mask, 0.0, x_full)
    # x_full = jnp.where(imposed_mask, imposed_vals_t, x_full)
    # return x_full
    x_full = jnp.zeros((free_mask.size,), dtype=fixed_vals_t.dtype)
    x_full = x_full.at[free_mask].set(x_free)
    x_full = jnp.where(fixed_mask, fixed_vals_t, x_full)
    x_full = jnp.where(imposed_mask, imposed_vals_t, x_full)
    return x_full


def _get_before_tip(tip_pos: jnp.ndarray,
                    tip_angle: jnp.ndarray,
                    L: float,
                    *,
                    dtype=jnp.float32) -> jnp.ndarray:
    """Return coordinates of the node that is one before the tip.

    tip_pos: (2,) tip [x, y]
    tip_angle: scalar (radians), CCW from +x
    L: edge length of last link
    """
    tip_pos = jnp.asarray(tip_pos, dtype=dtype).reshape((2,))
    dx = L * jnp.cos(tip_angle)
    dy = L * jnp.sin(tip_angle)
    return tip_pos - jnp.array([dx, dy], dtype=dtype)


def _get_tip_angle(pos_arr: np.array) -> np.array:
    """
    pos_arr: array of shape (H, 2)
    Returns: angle (radians) in [-pi, pi], measured from -x axis
    """
    pos_arr = np.asarray(pos_arr)
    p0, p1 = pos_arr[-2], pos_arr[-1]   # last two points
    dx, dy = p1 - p0                    # displacement vector

    # shift so that 0 is along -x
    theta_from_negx = np.arctan2(dy, dx) - np.pi
    # normalize back to [-pi, pi]
    theta_from_negx = (theta_from_negx + np.pi) % (2*np.pi) - np.pi

    return theta_from_negx


def torque(tip_angle: float, Fx: float, Fy: float) -> float:
    return np.cos(tip_angle)*Fy-np.sin(tip_angle)*Fx


def tip_torque(tip_angle: float, Forces: NDArray) -> float:
    Fy_last = Forces[-1]
    Fx_last = Forces[-2]
    Fy_before_last = Forces[-3]
    Fx_before_last = Forces[-4]
    return np.cos(tip_angle)*Fy_last-np.sin(tip_angle)*Fx_last - (np.cos(tip_angle)*Fy_before_last -
                                                                  np.sin(tip_angle)*Fx_before_last)


# ### NOT IN USE
#     @staticmethod
#     def _compute_thetas_over_traj(Strctr: "StructureClass", traj_pos: jax.Array) -> jax.Array:
#         """
#         Compute hinge angles (T,H) from a trajectory of positions (T,N,2)
#         using StructureClass.hinge_angle(pos, h). Returns radians.
#         """
#         H = Strctr.hinges
#         hinge_ids = jnp.arange(H, dtype=jnp.int32)

#         # vmaps: over time, then over hinges
#         per_time = jax.vmap(lambda P: jax.vmap(lambda h: Strctr.hinge_angle(P, h))(hinge_ids))
#         return per_time(traj_pos)  # (T,H)
