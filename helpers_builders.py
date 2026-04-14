from __future__ import annotations

import numpy as np
import copy
import jax
import jax.numpy as jnp
import re
import pandas as pd

from pathlib import Path
from collections import Counter, defaultdict
from jax import grad, jit, vmap
from typing import Tuple, List
from numpy import array, zeros
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional, Iterable, Mapping

if TYPE_CHECKING:
    from StructureClass import StructureClass
    from EquilibriumClass import EquilibriumClass
    from VariablesClass import VariablesClass

import file_funcs

np.set_printoptions(precision=4, suppress=True)


# ---------------------------------------------------------------
# Convenience: move from jax to numpy arrays and vice-verse
# ---------------------------------------------------------------
def jax2numpy(arr: jax.Array, dtype=float) -> NDArray[np.float64]:
    """
    Convert JAX array to NumPy array.

    Breaks JAX's tracing, so should only be used at the boundaries of your pipeline.
    """
    # return np.asarray(arr)
    if dtype == float:
        return np.asarray(jax.device_get(arr))
    else:
        return np.asarray(jax.device_get(arr), dtype=int)


def numpy2jax(arr: NDArray[np.float64]) -> jnp.ndarray:
    """
    Convert a NumPy array (or Python list) back to a JAX array.

    This creates a regular JAX array on the default device.
    """
    return jnp.asarray(arr)


# ---------------------------------------------------------------
# Reshapes
# ---------------------------------------------------------------
def _reshape_pos_arr_2_state(pos_arr: jnp.Array[jnp.float64]) -> jnp.Array[jnp.float64]:
    """
    Flatten position array into a full state vector ([x0, y0, x1, y1, .... x0_dot, y0_dot...]).

    Notes:
    -----
    - Not used as of 2026Mar

    Parameters
    ----------
    pos_arr : (nodes, 2) jnp.ndarray. Node positions, with each row containing (x, y) coordinates.

    Returns
    -------
    state : (2*N,) jnp.ndarray. Concatenated vector, 1st N*2 idxs = flattened pos, last N*2 = initialized velocities (all zeros).
    """
    first_half = pos_arr.flatten()
    second_half = jnp.zeros_like(first_half)
    return jnp.concatenate([first_half, second_half])


def _reshape_state_2_pos_arr(state: jnp.Array[jnp.float64], pos_arr: jnp.Array[jnp.float64]) -> jnp.Array[jnp.float64]:
    """
    Reshape a flattened state vector back into node positions.

    Parameters
    ----------
    state : (2*N,) jnp.ndarray. Flattened state vector of positions and velocities.
    pos_arr : (N, 2) jnp.ndarray. Template array defining the desired shape for positions.

    Returns
    -------
    pos_arr : (N, 2) jnp.ndarray. Reshaped positions, extracted from the first N*2 entries of `state`.
    """
    return state.reshape(pos_arr.shape)


# ---------------------------------------------------------------
# Chain index conventions
# ---------------------------------------------------------------
def dof_idx(node: int, comp: int) -> int:
    """Return the flat DOF index for a node and component.

    Parameters:
    -----------
    node - int, node number in {0,N}
    comp - int {0, 1}, comp = 0 → x, comp = 1 → y

    returns:
    --------
    DOF index, int
    """
    return 2 * node + comp


# ---------------------------------------------------------------
# Initiations
# ---------------------------------------------------------------
def _initiate_pos(nodes: int, L: float, numpify: bool = False) -> jax.Array:
    """
    `(hinges+2, 2)` each pair is (xi, yi) of point i going like [[0, 0], [1, 0], [2, 0], etc]
    
    Parameters
    ----------
    hinges  - int. Number of hinges. Number of nodes will be hinges + 2 (two end nodes + internal ones).
    L       - float. Length of edge/facet [m].
    numpify - bool. True = return numpy array, else: jax

    Returns
    -------
    pos_arr : jnp.ndarray, (hinges+2, 2), Node coordinates laid out on x-axis, starting from (0,0).
              Example for hinges=2: [[0,0], [1,0], [2,0], [3,0]]
    """
    flat = L*jnp.arange(nodes, dtype=jnp.float32)
    pos_arr = jnp.stack([flat, jnp.zeros_like(flat)], axis=1)
    if numpify:
        return jax2numpy(pos_arr)
    else:
        return pos_arr


def _initiate_buckle(hinges: int, shims: int, buckle_pattern: tuple = (), numpify: bool = False) -> jax.Array:
    """
    `(hinges, shims)` of +1 and -1 denoting buckle down and up, respectively, of every shims in every hinge.

    Parameters
    ----------
    hinges         - int. Number of hinges. Number of nodes will be hinges + 2 (two end nodes + internal ones).
    shims          - int. Number of shims per hinge. always 1.
    buckle_pattern - optional tuple (H,) of +1 and -1 denoting buckle state of shims (assuming one per hinge)
    numpify        - bool. True = return numpy array, else: jax

    Returns
    -------
    pos_arr - jnp.ndarray, (hinges, shims), +1 and -1, denoting buckle down and up, respectively, of every shims in every hinge.
    """
    buckle = -jnp.ones((hinges, shims))
    pattern = jnp.array(buckle_pattern)   # shape (H,)

    # bottom shim = index 1
    if jnp.shape(pattern) == ():
        if pattern == 1:
            buckle.at[0, 0].set(1)
    else:
        buckle = buckle.at[jnp.where(pattern == +1)[0], 0].set(1)

    if numpify:
        return jax2numpy(buckle)
    else:
        return buckle


# ---------------------------------------------------------------
# Physical helpers
# ---------------------------------------------------------------
def clamp_pos_same_delta(*, before_prev: NDArray, tip_angle_new: float, tip_raw: NDArray, second_node: NDArray, R_lim: float,
                         L: float, eps=1e-12):
    """
    Enforce ||node before tip - second_node|| <= R_lim
    while preserving ||new node before tip - before_prev|| = ||before_raw - before_prev|| (when possible).
    by solving a circle–circle intersection between:
      1) the constraint circle centered at ``second_node`` with radius ``R_lim``
      2) the "same-step" circle centered at ``before_prev`` with radius ``step``

    If intersections exist, the solution closest to ``before_raw`` is chosen. If not, a radial
    projection onto the constraint circle is used as a fallback.

    Notes
    -----
    - The "same step" constraint is enforced only when the two circles intersect.
      Otherwise, the solution is the closest feasible point obtained by radial projection.
    - When ``r_raw`` is extremely small (near the circle center), a deterministic fallback point
      on the constraint circle is used to avoid division by zero.
    - all sizes in [m]

    Parameters
    ----------
    before_prev   - np.array(float), (2,), Position of the node-before-tip at previous step.
    tip_angle_new - float. Updated tip angle (radians), measured CCW, used to infer the before-tip point.
    tip_raw       - np.array(float), (2,), Proposed updated tip position *before* applying radius clamp.
    second_node   - np.array(float), (2,), Reference point for radius constraint (typically second fixed node, e.g. ``[L, 0]``)
    R_lim         - float. Maximum allowed distance from second_node to node-before-tip. Calculated at effective_radius()
    L             - float. Link length between the tip node and the node-before-tip.
    eps           - float, optional, Small tolerance for numerical stability and comparisons.

    Returns
    -------
    tip_new : ndarray(float), (2,), clamped tip position consistent with tip_angle_new and the clamped before-tip point.
    before_new : ndarray(float), (2,) clamped node-before-tip position satisfying the radius constraint.
    clamped : bool. True if clamping was applied, False if ``before_raw`` already satisfied the constraint.
    """
    before_prev = np.asarray(before_prev, float).reshape(2,)
    tip_raw = np.asarray(tip_raw, float).reshape(2,)
    second_node = np.asarray(second_node, float).reshape(2,)
    R_lim = float(R_lim)

    # raw before-tip implied by tip_raw and tip_angle_new
    before_raw = tip_raw - L*array([np.cos(tip_angle_new), np.sin(tip_angle_new)], float)

    disp_raw = before_raw - second_node
    r_raw = np.linalg.norm(disp_raw)

    if r_raw <= R_lim + eps:
        return tip_raw, before_raw, False  # no clamp

    step = np.linalg.norm(before_raw - before_prev)

    # circle-circle intersection: constraint circle & step circle
    pts = _circle_circle_intersections_np(second_node, R_lim, before_prev, step, eps=eps)

    if len(pts) == 0:
        # fallback: radial clamp in before-space
        if r_raw < eps:
            before_new = second_node + array([R_lim, 0.0])
        else:
            before_new = second_node + disp_raw * (R_lim / r_raw)
    else:
        # pick intersection closest to raw proposal
        before_new = min(pts, key=lambda p: np.sum((p - before_raw)**2))

    tip_new = before_new + L*array([np.cos(tip_angle_new), np.sin(tip_angle_new)], float)
    return tip_new, before_new, True


def _circle_circle_intersections_np(c0: NDArray[np.float64], r0: float, c1: NDArray[np.float64], r1: float, eps=1e-12):
    """
    Intersection points of two circles in 2D (NumPy version). 
    - Circle 0: center ``c0`` and radius ``r0``
    - Circle 1: center ``c1`` and radius ``r1``
    this function returns their intersection point(s), if they exist.

    Parameters
    ----------
    c0 : np.array(float), (2,), Center of the first circle.
    r0 : float, Radius of the first circle.
    c1 : np.array(float), (2,), Center of the second circle.
    r1 : float, Radius of the second circle.
    eps : float, optional, Small tolerance for numerical stability and degeneracy checks.

    Returns
    -------
    pts : list of ndarray
        - ``[]``         → no intersection (separate circles, contained circle, or degenerate case)
        - ``[p]``        → one intersection point (tangent case)
        - ``[p1, p2]``   → two intersection points

    Notes
    -----
    - If the distance between centers is greater than ``r0 + r1`` (disjoint)
      or smaller than ``|r0 - r1|`` (one circle fully inside the other),
      no intersection is returned.
    - This function is used in geometric clamping routines (e.g. enforcing
      equal step size while respecting a radius constraint).
    """
    c0 = np.asarray(c0, float).reshape(2,)
    c1 = np.asarray(c1, float).reshape(2,)
    r0 = float(r0); r1 = float(r1)

    dvec = c1 - c0
    d = np.linalg.norm(dvec)

    if d > r0 + r1 + eps:
        return []
    if d < abs(r0 - r1) - eps:
        return []
    if d < eps:  # coincident centers / degenerate
        return []

    a = (r0*r0 - r1*r1 + d*d) / (2*d)
    h2 = r0*r0 - a*a
    if h2 < 0:
        h2 = 0.0
    h = np.sqrt(h2)

    p2 = c0 + a * dvec / d
    perp = array([-dvec[1], dvec[0]]) / d

    if h <= eps:
        return [p2]
    return [p2 + h*perp, p2 - h*perp]


def _correct_big_stretch(tip_pos: NDArray[np.float64], tip_angle: float, total_angle: float, R_free: float,
                         L: float, margin: float = 0.0, supress_prints: bool = True) -> NDArray[np.float64]:
    """
    Radially scale down tip position to maximal reachable radius constraint, if tip position exceeds it.
    Applied to distance between node-before-tip and 2nd node (located at (L, 0)). Scale down is radial towards 2nd node.


    Notes
    -----
    - Maximal allowed radius computed using R_eff = effective_radius(), accounting for coil wrap & geometric shrinkage.

    Parameters
    ----------
    tip_pos : ndarray (2,), Initially proposed tip position.
    tip_angle : float, Tip orientation (radians), measured CCW.
    total_angle : float, Unwrapped accumulated chain angle (can exceed ±2π).
    R_free : float, Nominal maximal free radius before shrink corrections.
    L : float, Edge/facet length.
    margin : float, optional, Additional safety factor (fraction of L) subtracted from effective radius.
    supress_prints : bool, optional, If False, prints diagnostic information.

    Returns
    -------
    tip_pos_corrected : ndarray, (2,), Corrected tip position if clamping applied, otherwise returns original tip_pos.
    """
    # Compute the location of the node before the tip
    # before_last = array([tip_pos[0] - L * np.cos(tip_angle), tip_pos[1] - L * np.sin(tip_angle)])
    before_last = _get_before_tip(tip_pos, tip_angle, L)

    # Second node from the base sits at (L, 0)
    second_node = array([L, 0.0], dtype=float)

    # chain current radius
    disp = before_last - second_node
    r_chain = np.hypot(disp[0], disp[1])
    R_eff = effective_radius(R_free, L, total_angle, tip_angle, supress_prints=supress_prints)

    if not supress_prints:
        print(f'update vals before correction={tip_pos},{tip_angle}')
        print(f'r_chain{r_chain}')
        print(f'R_eff{R_eff}')

    # # old before Mar17
    # x2, y2 = None, None

    # if r_chain >= (R_eff - margin*L):
    #     scale = (R_eff - margin*L) / r_chain
    #     x2 = second_node[0] + (tip_pos[0]-second_node[0]) * scale
    #     y2 = second_node[1] + (tip_pos[1]-second_node[1]) * scale
    #     if not supress_prints:
    #         print(f'clamped from x={tip_pos[0]},y={tip_pos[1]} to x={x2},y={y2}')
    #     return array([x2, y2])
    # else:
    #     return tip_pos

    # # new after Mar17
    if r_chain <= max(0.0, R_eff - margin*L):
        return tip_pos

    # clamp BEFORE-TIP, not TIP
    scale = (R_eff - margin*L) / r_chain
    before_new = second_node + disp * scale

    # reconstruct tip from clamped before-tip and prescribed tip angle
    tip_new = before_new + L * np.array([np.cos(tip_angle), np.sin(tip_angle)], dtype=float)

    if not supress_prints:
        print(f"clamped from x={tip_pos[0]},y={tip_pos[1]} "
              f"to x={tip_new[0]},y={tip_new[1]}")

    return tip_new


def effective_radius(R: float, L: float, total_angle: float, tip_angle: float, margin: float = 0.0, 
                     supress_prints: bool = True) -> float:
    """
    Compute effective maximal reachable radius of the chain, accounting for angular wrapping (coil shrinkage).

    Notes
    -----
    - `total_angle` must be **unwrapped**, i.e. it may exceed ±2π, ±4π, etc.
    - For every full revolution (2π) in `delta`, the effective radius
      shrinks by 2L, corresponding to a full loop consuming two edge lengths.
    - Remaining partial revolution contributes additional shrinkage = L * (1 - cos(rem / 2))

    Parameters
    ----------
    R           - float, Nominal free radius before accounting for coiling, calcaulted at init of SupervisorClass.
    L           - float, Edge length of the chain.
    total_angle - float, Unwrapped accumulated chain angle (radians).
    tip_angle   - float,  Current tip orientation (radians).
    margin      - float, optional,  Additional safety margin subtracted from R.
    supress_prints - bool, optional, If False, prints shrink contributions.

    Returns
    -------
    R_eff - float,  Effective maximal reachable radius after accounting for coil-induced shrinkage.
    """
    two_pi = 2.0 * np.pi

    # ------ tip ------
    delta = float(np.abs(total_angle - tip_angle))  # radians, unwrapped
    
    n_rev = int(np.floor(delta / two_pi))
    rem = delta - n_rev * two_pi  # in [0, 2π)

    shrink_full_tip = (2.0 * L) * n_rev
    
    shrink_partial_tip = L * (1.0 - np.cos(rem / 2.0))  # in [0, 2L)
    # shrink_partial = L * (1.0 - np.cos(rem))  # in [0, 2L)
    if not supress_prints:
        print('shrink due to full tip revolutions [mm]', shrink_full_tip)
        print('shrink due to partial tip revolution [mm]', shrink_partial_tip)

    # ------ total angle ------
    n_halfturns = int(np.floor((np.abs(total_angle) + np.pi) / (2.0 * np.pi)))
    shrink_full_total_angle = (1.0 * L) * n_halfturns
    if not supress_prints:
        print('shrink due to total angle revolutions around base [mm]', shrink_full_total_angle)

    shrink = shrink_full_tip + shrink_partial_tip + shrink_full_total_angle
    return max(0.0, (R - margin) - shrink)


def swept_last_edge_crosses_first_edge(before_prev: NDArray[np.float64], tip_prev: NDArray[np.float64],
                                       before_new: NDArray[np.float64], tip_new: NDArray[np.float64], L: float,
                                       *, eps: float = 1e-12, include_endpoints: bool = False) -> bool:
    """
    Return whether the quadrilateral swept by the last edge crosses the first edge.

    The swept quadrilateral is taken as:
        before_prev -> tip_prev -> tip_new -> before_new

    The first edge is:
        [ [0,0], [L,0] ]

    Parameters
    ----------
    before_prev, tip_prev, before_new, tip_new
        Endpoints of the old and new last edge, each shape (2,).
    L
        Edge length of the first segment.
    eps
        Numerical tolerance.
    include_endpoints
        Whether mere touching counts as crossing.

    Returns
    -------
    bool
        True if the first edge intersects the swept quadrilateral.
    """
    first_a = np.array([0.0, 0.0], dtype=float)
    first_b = np.array([float(L), 0.0], dtype=float)

    quad = np.array([before_prev, tip_prev, tip_new, before_new], dtype=float)

    # Check intersection with all 4 boundary edges of the swept quadrilateral
    for i in range(4):
        q0 = quad[i]
        q1 = quad[(i + 1) % 4]
        if _segments_intersect(first_a, first_b, q0, q1, eps=eps, include_endpoints=include_endpoints,):
            return True
    return False


def _origin_cut_side(before_prev: NDArray[np.float64], tip_prev: NDArray[np.float64],
                     before_new: NDArray[np.float64], tip_new: NDArray[np.float64]) -> float:
    """
    Determine from which side the swept last edge crosses the first edge.
    Negative -> from below, positive -> from above.
    """
    y_mean = 0.25 * (before_prev[1] + tip_prev[1] + before_new[1] + tip_new[1])

    # fallback if almost perfectly symmetric
    if abs(y_mean) < 1e-12:
        y_mean = 0.5 * (tip_prev[1] + tip_new[1])

    return float(np.sign(y_mean)) if abs(y_mean) > 1e-12 else 1.0


def evade_first_edge_by_sliding(tip_prev: NDArray[np.float64], delta_tip_raw: NDArray[np.float64],
                                L: float, *, eps: float = 1e-12) -> NDArray[np.float64]:
    """
    Replace an inward motion toward the first edge by a tangential motion
    around the nearest point on the first edge.

    Parameters
    ----------
    tip_prev
        Previous accepted tip position, shape (2,).
    delta_tip_raw
        Raw proposed tip displacement, shape (2,).
    L
        Length of the first edge, which lies on [(0,0), (L,0)].
    eps
        Small tolerance.

    Returns
    -------
    delta_tip_safe
        Modified displacement that slides tangentially around the first edge.
    """
    tip_prev = np.asarray(tip_prev, dtype=float).reshape(2,)
    delta_tip_raw = np.asarray(delta_tip_raw, dtype=float).reshape(2,)

    # Closest point on the first edge segment
    c = np.array([np.clip(tip_prev[0], 0.0, L), 0.0], dtype=float)

    r = tip_prev - c
    r_norm = np.linalg.norm(r)

    if r_norm < eps:
        # Degenerate case: exactly on the segment; pick horizontal escape
        return np.array([-np.linalg.norm(delta_tip_raw), 0.0], dtype=float)

    r_hat = r / r_norm

    # Split raw motion into radial + tangential parts
    radial_mag = np.dot(delta_tip_raw, r_hat)
    delta_rad = radial_mag * r_hat
    delta_tan_pref = delta_tip_raw - delta_rad

    # If already not moving inward, keep it
    if radial_mag >= 0.0:
        return delta_tip_raw

    # Build the two tangential directions
    t_ccw = np.array([-r_hat[1],  r_hat[0]], dtype=float)
    t_cw  = np.array([ r_hat[1], -r_hat[0]], dtype=float)

    # Choose tangent most aligned with intended tangential component
    if np.linalg.norm(delta_tan_pref) > eps:
        if np.dot(delta_tan_pref, t_ccw) >= np.dot(delta_tan_pref, t_cw):
            t_hat = t_ccw
        else:
            t_hat = t_cw
        tan_mag = np.linalg.norm(delta_tan_pref)
    else:
        # No tangential preference in raw motion:
        # choose the direction that moves away from the nearer endpoint
        if c[0] < 0.5 * L:
            t_hat = t_cw   # near origin -> tends to go left when below axis
        else:
            t_hat = t_ccw
        tan_mag = np.linalg.norm(delta_tip_raw)

    return tan_mag * t_hat


def coil(angle: float, revolutions: float = 1.5):
    """
    return boolean whether the tip coiled too much

    Parameters:
    -----------
    angle       - float, angle during update state after corrections 
    revolutions - float, how many 2pi revolution allowed before angle is considered as too much coiled

    Returns:
    --------
    boolean - True=coiled too much, correct inside SupervisorClass.calc_update
    """
    return np.abs(angle) > revolutions * 2*np.pi


# ---------------------------------------------------------------
# Geometrical helpers
# ---------------------------------------------------------------
def _get_before_tip(tip_pos: NDArray, tip_angle: float, L: float, *, xp=jnp, dtype=None):
    """
    Return coordinates of the node that is one before the tip given the position of the last node and tip angle

    Notes:
    ------
    Works with numpy (xp=np) or jax.numpy (xp=jnp).

    Parameters:
    -----------
    tip_pos   - np.array(float), (2,), position of last node
    tip_angle - float
    L         - float, edge/facet length [mm]

    Returns:
    --------
    np.array(float), (2,), position of before the last node
    """
    if dtype is None:
        tip_pos = xp.asarray(tip_pos).reshape((2,))
    else:
        tip_pos = xp.asarray(tip_pos, dtype=dtype).reshape((2,))

    dx = L * xp.cos(tip_angle)
    dy = L * xp.sin(tip_angle)

    if dtype is None:
        return tip_pos - xp.array([dx, dy])
    return tip_pos - xp.array([dx, dy], dtype=dtype)


def _get_total_angle(tip_pos: NDArray, prev_total_angle: float, L: float) -> NDArray:
    """
    angle between tip and last fixed node, CCW

    Parameters
    ----------
    tip_pos          - (2,) array, Current tip position.
    prev_total_angle - float, The accumulated unwrapped angle up to the previous timestep.
    L                - float, Edge length (used to define reference point at (L,0)).

    Returns
    -------
    new_total_angle : float, The unwrapped angle (can exceed ±pi, ±2pi, ±3pi, ...).
    """
    # ------ total angle [-pi/2, pi/2] ------
    dx, dy = array([L, 0]) - tip_pos  # displacement vector

    # shift so that 0 is along -x
    total_angle = np.arctan2(dy, dx) - np.pi
    # normalize back to [-pi, pi]
    total_angle = (total_angle + np.pi) % (2*np.pi) - np.pi

    # ------ correct for wrapping around center ------
    prev_theta_wrapped = ((prev_total_angle + np.pi) % (2*np.pi)) - np.pi
    delta = total_angle - prev_theta_wrapped

    # correct jumpt across -x axis - adding or subtracting 2π
    if delta > np.pi:
        delta -= 2*np.pi
    elif delta < -np.pi:
        delta += 2*np.pi

    # Update cumulative angle
    total_angle = prev_total_angle + delta

    return total_angle


def _get_tip_angle(pos_arr: NDArray) -> NDArray:
    """
    angle of edge connected to tip relative to horizontal, CCW

    Parameters:
    -----------
    pos_arr - np.array, (H, 2), chain node positions

    Returns:
    --------
    angle (radians) in [-pi, pi], measured from -x axis, CCW
    """
    pos_arr = np.asarray(pos_arr)
    p0, p1 = pos_arr[-2], pos_arr[-1]   # last two nodes
    dx, dy = p1 - p0                    # displacement vector

    # shift so that 0 is along -x
    # theta_from_negx = np.arctan2(dy, dx) - np.pi

    # shift so that 0 is along +x
    theta_from_negx = np.arctan2(dy, dx)
    # normalize back to [-pi, pi]
    theta_from_negx = (theta_from_negx + np.pi) % (2*np.pi) - np.pi

    return theta_from_negx
    @staticmethod
    def _compute_thetas_over_traj(Strctr: "StructureClass", traj_pos: jax.Array) -> jax.Array:
        """
        Compute hinge angles (T,H) from a trajectory of positions (T,N,2)
        using StructureClass.hinge_angle(pos, h). Returns radians.
        """
        H = Strctr.hinges
        hinge_ids = jnp.arange(H, dtype=jnp.int32)

        # vmaps: over time, then over hinges
        per_time = jax.vmap(lambda P: jax.vmap(lambda h: Strctr.hinge_angle(P, h))(hinge_ids))
        return per_time(traj_pos)  # (T,H)


def _point_segment_closest(p: jax.array, a: jax.array, b: jax.array, eps: float = 1e-12):
    """
    Compute the closest point on a line segment to a given point.

    Given a point `p` and a line segment defined by endpoints `a` and `b`, finds the point `c` on the segment `[a, b]` that is
    closest to `p`, and returns the displacement vector from `c` to `p`, its squared norm, and segment interpolation parameter.

    Parameters
    ----------
    p   - jax.Array, shape (2,), Query point.
    a   - jax.Array, shape (2,), First endpoint of the segment.
    b   - jax.Array, shape (2,), Second endpoint of the segment.
    eps - float, optional, Small positive constant added to the denominator to avoid division by zero

    Returns
    -------
    d  - jax.Array, (2,), Displacement vector from the closest point on the segment to `p`: d = p - c
    d2 - jax.Array, scalar, Squared distance from `p` to the segment: d2 = ||p - c||²
    t  - jax.Array, scalar,  Segment interpolation parameter in [0, 1] locating the closest point: c = a + t (b - a)
    """
    ab = b - a
    ap = p - a
    denom = jnp.dot(ab, ab) + eps
    t = jnp.clip(jnp.dot(ap, ab) / denom, 0.0, 1.0)
    c = a + t * ab
    d = p - c
    d2 = jnp.dot(d, d)
    return d, d2, t  # d = (p-c)


def _on_segment(p: NDArray[np.float64], q: NDArray[np.float64], r: NDArray[np.float64], *, eps: float = 1e-12) -> bool:
    """
    Return whether q lies on the closed segment [p, r].
    """
    return (min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps and min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps)


def _segments_intersect(a: NDArray[np.float64], b: NDArray[np.float64], c: NDArray[np.float64], d: NDArray[np.float64],
                        *, eps: float = 1e-12, include_endpoints: bool = True) -> bool:
    """
    Return whether the closed segments [a,b] and [c,d] intersect.

    Parameters
    ----------
    a, b, c, d
        Segment endpoints, each shape (2,).
    eps
        Numerical tolerance.
    include_endpoints
        Whether touching at endpoints / collinear touching counts as intersection.

    Returns
    -------
    bool
        True if the segments intersect.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)

    o1 = _orient(a, b, c)
    o2 = _orient(a, b, d)
    o3 = _orient(c, d, a)
    o4 = _orient(c, d, b)

    # Proper crossing
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and \
       ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
        return True

    if not include_endpoints:
        return False

    # Touching / collinear cases
    if abs(o1) <= eps and _on_segment(a, c, b, eps=eps):
        return True
    if abs(o2) <= eps and _on_segment(a, d, b, eps=eps):
        return True
    if abs(o3) <= eps and _on_segment(c, a, d, eps=eps):
        return True
    if abs(o4) <= eps and _on_segment(c, b, d, eps=eps):
        return True

    return False


def _orient(p: NDArray[np.float64], q: NDArray[np.float64], r: NDArray[np.float64]) -> float:
    """
    Signed 2D orientation / cross product of pq with pr.

    Returns
    -------
    float
        > 0 for counter-clockwise turn,
        < 0 for clockwise turn,
        = 0 for collinear points.
    """
    return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))


# ---------------------------------------------------------------
# DOFs - free and essential - jax instances
# ---------------------------------------------------------------
def _assemble_full_from_free(free_mask: jax.Array, fixed_mask: jax.Array, imposed_mask: jax.Array, x_free: jax.Array,
                             fixed_vals_t: jax.Array, imposed_vals_t: jax.Array) -> jax.Array:
    """
    Build full flattened x from free DOFs + constraints at time t.
    organized as (x0, y0, x1, y2, ..., xN, yN).

    Parameters
    ----------
    x_free         - jnp.ndarray (nodes_free*2,), Values of free degrees of freedom.
    free_mask      - jnp.ndarray[bool], (nodes*2,) Boolean mask marking which entries are free DOFs.
    fixed_mask     - jnp.ndarray[bool], (nodes*2,) Boolean mask marking fixed DOFs (always zero).
    imposed_mask   - jnp.ndarray[bool] (nodes*2,) Boolean mask marking imposed DOFs (prescribed values).
    imposed_vals_t - jnp.ndarray (nodes*2,), Values to assign to imposed DOFs at time t.

    Returns
    -------
    x_full : jnp.ndarray (n_coords,), Complete flattened coordinate vector, consistent with all masks and imposed values.
    """
    x_full = jnp.zeros((free_mask.size,), dtype=fixed_vals_t.dtype)
    x_full = x_full.at[free_mask].set(x_free)
    x_full = jnp.where(fixed_mask, fixed_vals_t, x_full)
    x_full = jnp.where(imposed_mask, imposed_vals_t, x_full)
    return x_full


def _extend_pos_to_x0_v0(init_pos: NDArray, pos_noise: Optional[float], vel_noise: Optional[float], rand_key: int) -> NDArray:
    """
    Build full state vector [x, v] from positions, optionally adding noise to interior DOFs only.
    where x = [x0, y0, x1, y1, ..., xN, yN] and v = [vx0, vy0, ..., vxN, vyN] (which are all zeros)

    Notes
    -----
    - used at beginning of equilibration in EquilibriumClass. Initialization prior to ODE integration.
    - Noise is sampled independently per DOF from -U[-1, 1] and scaled by ``pos_noise`` or ``vel_noise``.
    - The same PRNG seed is currently used for both position and velocity noise.

    Parameters
    ----------
    init_pos  - jax.Array (nodes, 2), Initial nodal positions.
    pos_noise - float or None. Amplitude of position noise per interior DOF. If None, no positional noise is added.
    vel_noise - float or None. Amplitude of velocity noise per interior DOF. If None, velocities remain zero.
    rand_key - int. Integer seed used to initialize a JAX PRNGKey for reproducible noise.

    Returns
    -------
    state_0 - jax.Array (2 * n_coords,) (=2*2*N). Concatenated flattened position and velocity vector. Velocities are zero
    """
    # start from current geometry
    x0 = init_pos.flatten()  # (n_coords,)
    n_nodes = int(len(x0)/2)
    last = n_nodes-1
    # velocities: start at rest
    v0 = jnp.zeros_like(x0)

    # (a) Add positional noise only to interior nodes (exclude nodes 0,1,last-1,last)
    if pos_noise is not None or vel_noise is not None:
        # mask True for DOFs that *can* get noise, False for boundary nodes
        noise_mask = jnp.ones_like(x0, dtype=bool)
        boundary_nodes = (0, 1, last - 1, last)

        for node in boundary_nodes:
            if 0 <= node < n_nodes:
                noise_mask = noise_mask.at[dof_idx(node, 0)].set(False)
                noise_mask = noise_mask.at[dof_idx(node, 1)].set(False)

        if pos_noise is not None:
            # uniform noise in [-1, 1] per DOF
            key = jax.random.PRNGKey(rand_key)  # for reproducibility; swap to a passed-in key if needed
            rand = jax.random.uniform(key, shape=x0.shape, minval=-1.0, maxval=1.0)

            # apply scaled noise only where noise_mask is True
            x0 = x0 + (pos_noise * rand) * noise_mask.astype(x0.dtype)

        if vel_noise is not None:
            # uniform noise in [-1, 1] per DOF
            key = jax.random.PRNGKey(rand_key)  # for reproducibility; swap to a passed-in key if needed
            rand = jax.random.uniform(key, shape=v0.shape, minval=-1.0, maxval=1.0)

            # apply scaled noise only where noise_mask is True
            v0 = v0 + (vel_noise * rand) * noise_mask.astype(v0.dtype) 
    return jnp.concatenate([x0, v0], axis=0)


def _get_state_free_from_full(state_0: NDArray, fixed_mask: NDArray, imposed_mask: NDArray) -> Tuple[NDArray, int, NDArray]:
    """
    Extract free degrees of freedom (positions + velocities) from full flattened state vector state_0 = [x_flat, v_flat]
    x_flat - (n_coords,) = (x0, y0, ..., xN, yN),  
    v_flat - (n_coords,) = (vx0, vy0, ..., vxN, vyN)
    matching the convention used in `solve_dynamics`

    Parameters
    ----------
    state_0      - (2 * n_coords,) jax.Array, full flattened state vector [x0_flat, v0_flat]
    fixed_mask   - (n_coords,) jax.Array[bool], boolean mask marking position DOFs that are fixed.
    imposed_mask - (n_coords,) jax.Array[bool], Boolean mask marking position DOFs that are externally imposed.

    Returns
    -------
    free_mask   - jax.Array[bool] (n_coords,), Boolean mask marking position DOFs that are free.
    n_free_DOFs - int, Number of free **position** DOFs.
    state_free  - jax.Array (2 * n_free_DOFs,), Reduced flattened state vector containing:

    Notes
    -----
    - This function is used before ODE integration to construct the
      reduced system that evolves only over free DOFs.
    """
    n_coords = int(len(state_0)/2)
    free_mask = jnp.logical_not(imposed_mask | fixed_mask)
    n_free_DOFs = jnp.sum(free_mask)

    state_0 = state_0.flatten()
    state_0_x, state_0_x_dot = state_0[:n_coords], state_0[n_coords:]
    state_0_x_free, state_0_x_dot_free = state_0_x[free_mask], state_0_x_dot[free_mask]
    return free_mask, n_free_DOFs, jnp.concatenate([state_0_x_free, state_0_x_dot_free])


def _get_first_in_file(r: Mapping[str, Union[str, float, int, None]], keys: Iterable[str], *, name: str = "",
                       allow_missing: bool = False) -> Optional[tuple(float, str)]:
    """
    Extract first valid scalar value from a csv, using list of candidate keys. If no valid key is found: returns `None`

    Parameters
    ----------
    r             - Mapping[str, str | float | int | None]. Record-like object (e.g. CSV row dictionary).
    keys          - Iterable[str]. Ordered candidate keys to search for.
    name          - str, optional.  Human-readable field name used only for error reporting.
    allow_missing - bool, optional. If True, return None when no key is found.

    Returns
    -------
    value - float or None. First successfully parsed scalar value.
    """
    for k in keys:
        if k in r and r[k] not in ("", None):
            return float(r[k]), k
    if allow_missing:
        return None, None
    raise KeyError(f"None of {keys} found for {name}")


def _get_scalar_in_orthogonal_dir(vec: NDArray[np.floating], angle: float) -> float:
    """
    Compute scalar projection of vector onto direction orthogonal angle.

    Notes
    -----
    - The orthogonal direction is defined as n = [-sin(angle), cos(angle)] corresponding to +90° rotation of unit vector
      `[cos(angle), sin(angle)]`.

    Parameters
    ----------
    vec : NDArray, (2,) , vector whose orthogonal component is extracted.
    angle : float, Reference direction angle in radians.

    Returns
    -------
    scalar : float, Scalar projection of `vec` onto the orthogonal direction.
    """
    return -vec[0]*np.sin(angle) + vec[1]*np.cos(angle)


# -------------------------------------------------------------------
# Buckle helpers
# -------------------------------------------------------------------
# def sort_buckle_columns(cols: list[str]) -> list[str]:
#     """
#     Sort columns like:
#         buckle_h0_s0, buckle_h1_s0, ...
#     """
#     pat = re.compile(r"buckle_h(\d+)_s(\d+)")
#     parsed = []
#     for c in cols:
#         m = pat.fullmatch(c)
#         if m is not None:
#             h, s = map(int, m.groups())
#             parsed.append((h, s, c))
#     parsed.sort()
#     return [c for _, _, c in parsed]


def infer_buckle_columns(df: pd.DataFrame) -> list[str]:
    buckle_cols = [c for c in df.columns if re.fullmatch(r"buckle_h\d+_s\d+", c)]
    # buckle_cols = sort_buckle_columns(buckle_cols)
    if not buckle_cols:
        raise ValueError("No buckle_h*_s* columns found in CSV")
    return buckle_cols


# def build_transition_counts(folder: Path, only_init_and_final_buckles: bool = False):
#     """
#     Go over all final_loss_*.csv files and extract directed buckle transitions.

#     Returns
#     -------
#     transitions : Counter[(src, dst)] = number of times observed across all files
#     per_file_transitions : dict[file_name, list[(src, dst)]]
#     """
#     transitions = Counter()
#     per_file_transitions = {}
#     per_file_loss = {}

#     files = sorted(folder.glob("final_loss_*.csv"))
#     if not files:
#         raise FileNotFoundError(f"No files matching 'final_loss_*.csv' in {folder}")

#     for file in files:
#         df = pd.read_csv(file)
#         buckle_cols = infer_buckle_columns(df)

#         states = []
#         for _, row in df[buckle_cols].iterrows():
#             state = buckle_to_index(row.to_numpy())
#             states.append(state)

#         # keep only actual changes
#         edges_this_file = []
#         if only_init_and_final_buckles:
#             zip_states = zip(states[:1], states[-1:])
#         else:
#             zip_states = zip(states[:-1], states[1:])
#         for a, b in zip_states:
#             if a != b:
#                 edges_this_file.append((a, b))
#                 transitions[(a, b)] += 1

#         per_file_transitions[file.name] = edges_this_file

#         per_file_loss[file.name] = file_funcs.loss_from_filename(file)

#     return transitions, per_file_transitions, per_file_loss

def hamming_distance_int(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def build_transition_counts(folder: Path, only_init_and_final_buckles: bool = False, omit_inverted: bool = False):
    """
    Go over all final_loss_*.csv files and extract directed buckle transitions.

    Parameters
    ----------
    folder                      : path, all csv run files, from every init to every desired
    only_init_and_final_buckles : bool, True = transition is only from initial to final (not necessarily the desired)
                                  desired transition colored Cyan, undesired colored purple
    omit_inverted               : bool, True = do not account for  "_inverted.csv" output files

    Returns
    -------
    transitions          : Counter[(src, dst)] = number of times observed across all files
    per_file_transitions : dict[file_name, list[(src, dst)]]
    per_file_loss        : dict[file_name, float]
    edge_zero_loss_count : Counter[(src, dst)] = number of zero-loss files on this edge
    """
    transitions = Counter()
    per_file_transitions = {}
    per_file_loss = {}
    edge_zero_loss_count = Counter()  # all zeros initially

    files = sorted(folder.glob("final_loss_*.csv"))
    if not files:
        raise FileNotFoundError(f"No files matching 'final_loss_*.csv' in {folder}")
    if omit_inverted:  # neglect all files ending with "_inverted.csv"
        files = [f for f in files if not f.name.endswith("_inverted.csv")]

    for file in files:
        df = pd.read_csv(file)
        buckle_cols = infer_buckle_columns(df)

        states = []
        for _, row in df[buckle_cols].iterrows():
            state = buckle_to_index(row.to_numpy())
            states.append(state)

        loss = file_funcs.loss_from_filename(file)
        if only_init_and_final_buckles:
            per_file_loss[file.name] = loss

        edges_this_file = []
        if only_init_and_final_buckles:
            zip_states = zip(states[:1], states[-1:])
        else:
            zip_states = zip(states[:-1], states[1:])

        for a, b in zip_states:
            if a != b:
                edge = (a, b)
                edges_this_file.append(edge)
                transitions[edge] += 1
                if loss <= 1e-6 and only_init_and_final_buckles:
                    edge_zero_loss_count[edge] += 1
                # else:
                #     edge_zero_loss_count[edge] = 0

        per_file_transitions[file.name] = edges_this_file

    return transitions, per_file_transitions, per_file_loss, edge_zero_loss_count


def all_binary_states(n_bits: int) -> list[str]:
    return [format(i, f"0{n_bits}b") for i in range(2**n_bits)]


def all_possible_transitions(n_bits: int):
    n_states = 2 ** n_bits
    return [(i, j) for i in range(n_states) for j in range(n_states) if i != j]


def state_positions(n_bits: int, dx: float = 0.175, dy: float = 0.22,
                    x_margin: float = 0.065, y_margin: float = 0.05) -> dict[str, tuple[float, float]]:
    """
    Arrange states in layers by Hamming weight:
    0000 at top, 1111 at bottom.
    """
    # for k in range(n_bits + 1):
    #     layer = [s for s in all_binary_states(n_bits) if s.count("1") == k]
    #     layer.sort()
    #     x0 = -dx * (len(layer) - 1) / 2
    #     y = -dy * k
    #     for i, s in enumerate(layer):
    #         pos[s] = (x0 + i * dx, y)

    layers = []
    max_in_layer = 0

    for k in range(n_bits + 1):
        layer = [s for s in all_binary_states(n_bits) if s.count("1") == k]
        layer.sort()
        layers.append(layer)
        max_in_layer = max(max_in_layer, len(layer))

    # center every layer relative to the widest layer
    x_center = x_margin + dx * (max_in_layer - 1) / 2

    pos = {}
    for k, layer in enumerate(layers):
        width = dx * (len(layer) - 1) / 2
        x0 = x_center - width
        y = y_margin + dy * k
        for i, s in enumerate(layer):
            pos[s] = (x0 + i * dx, y)
    return pos


# ---------------------------------------------------------------
# Strings
# ---------------------------------------------------------------
def buckle_to_index(arr: NDArray) -> NDArray:
    """
    Convert [-1,1,1,-1] → integer index 0..15
    (-1 -> 0 , +1 -> 1)
    """
    bits = [(1 if x == 1 else 0) for x in arr]
    return bits[0]*8 + bits[1]*4 + bits[2]*2 + bits[3]


def index_to_buckle(i: int, n_bits: int = 4) -> str:
    """0 -> '0000', 15 -> '1111'"""
    return format(i, f"0{n_bits}b")


# # ==========
# # NOT IN USE
# # ==========

# def clamp_tip_no_cross(tip_pos, tip_angle, L, it=30):
#     """
#     used previously if correct_for_cut_origin==True
#     """
#     p0, p1 = array([0., 0.]), array([L, 0.])
#     s = array([L, 0.])
#     b = np.asarray(tip_pos, float)
#     d = b - s
#     if not inter(b, L, p0, p1, tip_angle): 
#         return b
# 
#     lo, hi = 0., 1.
#     for _ in range(it): 
#         mid = (lo+hi)/2
#         x = s+mid*d
#         (hi := mid) if inter(x) else (lo := mid)
#     return s + lo*d

# def inter(x, L, p0, p1, tip_angle, eps=1e-12):
#     """
#     Only used in clamp_tip_no_cross, which is unused
#     """
#     u = x - array([L*np.cos(tip_angle), L*np.sin(tip_angle)])
#     o = lambda A, B, C: (B[0]-A[0])*(C[1]-A[1])-(B[1]-A[1])*(C[0]-A[0])
#     o1, o2, o3, o4 = o(u, x, p0), o(u, x, p1), o(p0, p1, u), o(p0, p1, x)
#     return (o1*o2 < -eps) and (o3*o4 < -eps)

# def _correct_big_stretch(tip_pos: NDArray[np.float64], tip_angle: float, total_angle: float, L: float,
#                          edges: int) -> NDArray[np.float64]:
#     """
#     Used previously before _correct_big_stretch_robot_style
#     Physical upper bound on total chain stretch by correcting tip position to not exceed  maximal possible length.

#     Parameters
#     ----------
#     tip_pos : ndarray of shape (2,) previous tip position
    
#     tip_angle : float

#     Returns
#     -------
#     corrected_tip_pos : ndarray of shape (2,)
#         If the suggested tip position exceeds the chain's maximum reach, returns corrected position, scaled to maximal
#         physical stretch. Otherwise, the original tip_pos is returned.

#     Notes
#     -----
#     - before last node is located at:
#           before = tip_pos - L * [cos(theta), sin(theta)]
#     - Node 2 (second node) is located at (L, 0) in the reference frame.
#     - The correction preserves the direction of the displacement but 
#       rescales its magnitude down to the physical limit.
#     """

#     # Compute the location of the node before the tip
#     before_last = np.array([tip_pos[0] - L * np.cos(tip_angle), tip_pos[1] - L * np.sin(tip_angle)])

#     # Second node from the base sits at (L, 0)
#     second_node = np.array([L, 0.0], dtype=float)

#     # Actual stretch beyond the first two edges
#     vec = before_last - second_node
#     actual_stretch = np.linalg.norm(vec)

#     # Maximum physically possible stretch (remaining edges)
#     epsilon = 0.1*L  # add some breathing room

#     # wrap_tip = np.floor(np.abs(tip_angle) / np.pi)
#     wrap_tip = np.floor(np.abs(tip_angle-total_angle) / np.pi)
#     wrap_total = np.floor(np.abs(total_angle) / np.pi)
#     n_wrap = wrap_tip + wrap_total + 2

#     max_stretch = edges * L - n_wrap * L - epsilon
#     max_stretch = max(max_stretch, 0.0)

#     # If unphysical → scale back
#     if actual_stretch > max_stretch:
#         ratio = max_stretch / actual_stretch
#         corrected = (before_last - second_node) * ratio + (second_node + tip_pos - before_last)
#         return corrected

#     # Otherwise nothing to correct
#     return tip_pos

# def torque(tip_angle: float, Fx: float, Fy: float, L: float) -> float:
#     """
#     total torque
    
#     tip_angle - float, angle of tip of chain
#     Fx        - force [mN] in global x direction
#     Fy        - force [mN] in global y direction
#     """
#     F_orthogonal = _get_scalar_in_orthogonal_dir(array([Fx, Fy]), tip_angle) 
#     return F_orthogonal * L


# def tip_torque(tip_angle: float, Forces: NDArray) -> float:
#     F_xy_last = array([Forces[-2], Forces[-1]])
#     F_xy_before_last = array([Forces[-4], Forces[-3]])
#     last_orthogonal = _get_scalar_in_orthogonal_dir(F_xy_last, tip_angle)
#     before_last_orthogonal = _get_scalar_in_orthogonal_dir(F_xy_before_last, tip_angle)
#     return last_orthogonal - before_last_orthogonal
