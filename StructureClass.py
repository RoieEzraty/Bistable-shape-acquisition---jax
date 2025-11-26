from __future__ import annotations
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from typing import Tuple, List
from numpy import array, zeros
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional
from typing import Optional

import learning_funcs, helpers_builders


# ===================================================
# Class - Structure Variables - arrays hinge connections, edges, etc.
# ===================================================


class StructureClass(eqx.Module):
    """Bistable buckle structure (1D chain in the plane)."""  
    
    # ------ user-provided (static) ------
    hinges: int = eqx.field(static=True)     # number of hinges in the chain
    shims: int = eqx.field(static=True)      # shims per hinge
    L: float = eqx.field(static=True)        # rest length of rods

    # ------ computed in __init__ (static topology/geometry) ------
    edges_arr: NDArray[int] = eqx.field(init=False, static=True)          # (hinges+1, 2) int32
    edges: int = eqx.field(init=False, static=True)
    nodes: int = eqx.field(init=False, static=True)
    n_coords: int = eqx.field(init=False, static=True)
    hinges_arr: NDArray[int] = eqx.field(init=False, static=True)          # (hinges, 2) int32
    rest_lengths: NDArray[np.float_] = eqx.field(init=False, static=True)  # (hinges+1,) float32

    # ------ for equilibrium calculation, jax arrays ------
    fixed_mask: jax.Array[bool] = eqx.field(static=True)                   # (2*nodes,) float32

    # ------ optional learning graph (only if you call _build_learning_parameters) ------
    DM: Optional[NDArray[int]] = eqx.field(default=None, init=False, static=True)
    NE: Optional[NDArray[int]] = eqx.field(default=None, init=False, static=True)
    NN: Optional[NDArray[int]] = eqx.field(default=None, init=False, static=True)
    output_nodes_arr: Optional[NDArray[int]] = eqx.field(default=None, init=False, static=True)

    def __init__(self, hinges: int, shims: int, L: float, rest_lengths:  Optional[NDArray[np.float_]] = None,
                 update_scheme: str = 'one_to_one', Nin: Optional[int] = None, Nout: Optional[int] = None,
                 control_first_edge: Optional[bool] = True, control_tip_pos: Optional[bool] = True,
                 control_tip_angle: Optional[bool] = True):
        self.hinges = int(hinges)
        self.shims = int(shims)
        self.L = float(L)

        self.edges_arr = self._build_edges()            # (E=hinges+1, 2)
        self.edges = int(self.edges_arr.shape[0])
        self.nodes = self.edges + 1
        self.n_coords = self.nodes * 2 
        self.hinges_arr = self._build_hinges()           # (H=hinges, 2)
        self.rest_lengths = self._build_rest_lengths(rest_lengths=rest_lengths)  # rest lengths (float32)
        if update_scheme == 'BEASTAL':
            self.DM, self.NE, self.NN, self.output_nodes_arr = self._build_learning_parameters(Nin, Nout)
        self.fixed_mask = self._build_fixed_mask(control_first_edge)

        # learning fields left as None until _build_learning_parameters is called
       
    # --- builders ---
    def _build_edges(self) -> np.array[int]:
        starts = np.arange(self.hinges + 1, dtype=np.int32)
        return np.stack([starts, starts + 1], axis=1)

    def _build_hinges(self) -> np.array[int]:
        if self.hinges <= 0:
            return np.empty((0, 2), dtype=np.int32)
        starts = np.arange(self.hinges, dtype=np.int32)
        return np.stack([starts, starts + 1], axis=1)

    def _build_rest_lengths(self, rest_lengths: Optional[np.array[np.float_]]) -> np.array[np.float_]:
        if rest_lengths is not None:
            rl = np.asarray(rest_lengths, np.float32)
            assert rl.shape == (self.edges,), f"rest_lengths shape {rl.shape} != ({self.edges},)"
            return rl
        return jnp.full((self.edges,), self.L, dtype=np.float32)

    def _build_learning_parameters(self, Nin: int, Nout: int) -> None:
        _, _, _, DM, NE, NN, output_nodes_arr = learning_funcs.build_incidence(Nin, Nout)
        return DM, NE, NN, output_nodes_arr

    def _build_fixed_mask(self, control_first_edge: bool = True) -> None:
        # --- fixed and imposed DOFs initialize --- 
        fixed_mask = jnp.zeros((self.n_coords,), dtype=bool)

        # --- fixed: node 0, potentially also node 1 ---
        if control_first_edge:
            nodes = [0, 1]  # two nodes are at 0
        else:
            nodes = [0]     # first node at 0, 0

        for node in nodes:
            fixed_mask = fixed_mask.at[helpers_builders.dof_idx(node, 0)].set(True)
            fixed_mask = fixed_mask.at[helpers_builders.dof_idx(node, 1)].set(True)
        return fixed_mask

    # --- numpy geometry --- 
    def all_edge_lengths(self, pos_arr: NDArray[np.float_]) -> NDArray[np.float_]:
        vecs = pos_arr[self.edges_arr[:, 1]] - pos_arr[self.edges_arr[:, 0]]
        return np.linalg.norm(vecs, axis=1)  # shape: (self.edges,)
        # return jax.vmap(lambda e: self._get_edge_length(pos, e))(jnp.arange(self.edges))

    def all_hinge_angles(self, pos_arr: NDArray[np.float_]) -> NDArray[np.float_]:
        edge_nodes = self.edges_arr[self.hinges_arr]                          # (H, 2, 2)
        pts = pos_arr[edge_nodes]                                  # (H, 2, 2, 2)

        vecs = pts[..., 1, :] - pts[..., 0, :]                    # (H, 2, 2)
        u, v = vecs[:, 0, :], vecs[:, 1, :]                       # (H, 2), (H, 2)

        dot = np.sum(u * v, axis=-1)                              # (H,)
        cross_z = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]   # (H,)
        return np.arctan2(cross_z, dot).astype(np.float32)
        # return jax.vmap(lambda h: self._get_theta(pos, h))(jnp.arange(self.hinges))
     
    # --- jax geometry ---         
    def _normalize(self, v, eps=1e-9):
        n = jnp.linalg.norm(v)
        n_safe = jnp.maximum(n, eps)
        return v / n_safe

    def _angle_from_uv(self, u, v):
        un = self._normalize(u)
        vn = self._normalize(v)
        dot   = jnp.clip(jnp.dot(un, vn), -1.0, 1.0)  # not strictly needed for atan2, but helps near ±1
        cross = un[0]*vn[1] - un[1]*vn[0]
        return jnp.arctan2(cross, dot)

    def _get_theta(self, pos_arr: jax.Array, hinge: int) -> jax.Array:
        """Angle at a hinge (radians), CCW positive."""
        edges = jnp.asarray(self.edges_arr)
        hinges = jnp.asarray(self.hinges_arr)
        e0, e1  = hinges[hinge]
        i0, i1  = edges[e0]
        j0, j1  = edges[e1]
        u = pos_arr[i1] - pos_arr[i0]
        v = pos_arr[j1] - pos_arr[j0]
        return self._angle_from_uv(u, v)
        # edges = jnp.asarray(self.edges_arr)
        # hinges = jnp.asarray(self.hinges_arr)
        # e0, e1 = hinges[hinge]
        # i0, i1 = edges[e0]
        # j0, j1 = edges[e1]
        # u = pos_arr[i1] - pos_arr[i0]
        # v = pos_arr[j1] - pos_arr[j0]
        # dot = jnp.dot(u, v)
        # cross = u[0] * v[1] - u[1] * v[0]
        # return jnp.arctan2(cross, dot)            # shape: ()

    def _get_edge_length(self, pos_arr: jax.Array, edge: int) -> jax.Array:
        """Length of one edge given current positions pos: (Npoints,2) float."""
        edges = jnp.asarray(self.edges_arr)
        i, j = edges[edge]
        return jnp.linalg.norm(pos_arr[j] - pos_arr[i])
        # twopoints = pos_arr[self.edges_arr[edge]]  # (2,)
        # vec = twopoints[1, :] - twopoints[0, :]  # (2,)
        # return np.linalg.norm(vec)
