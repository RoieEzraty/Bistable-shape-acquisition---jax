from __future__ import annotations
import equinox as eqx
import jax
import jax.numpy as jnp

from typing import Tuple, List
from numpy import array, zeros
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional

import learning_funcs


# ===================================================
# Class - Structure Variables - arrays hinge connections, edges, etc.
# ===================================================


class StructureClass(eqx.Module):
    """Bistable buckle structure (1D chain in the plane)."""
    
    # --- user-provided (static) ---
    hinges: int = eqx.field(static=True)   # number of hinges in the chain
    shims: int = eqx.field(static=True)   # e.g. shim count per hinge (kept static)
    L: float = eqx.field(static=True)  # rest length of rods

    # --- computed in __init__ ---
    edges_arr:  jax.Array = eqx.field(init=False)   # (hinges+1, 2) point indices
    edges: int = eqx.field(init=False)
    nodes: int = eqx.field(init=False)
    n_coords: int = eqx.field(init=False)
    hinges_arr: jax.Array = eqx.field(init=False)   # (hinges, 2)  edge indices
    rest_lengths: jax.Array = eqx.field(init=False)  # (H+1,)   floats

    def __init__(self, hinges: int, shims: int, L: float, rest_lengths:  Optional[jax.Array] = None):
        self.hinges = int(hinges)
        self.shims = int(shims)
        self.L = float(L)

        self.edges_arr = self._build_edges()            # (E=hinges+1, 2)
        self.edges = jnp.shape(self.edges_arr)[0]
        self.nodes = self.edges + 1
        self.n_coords = self.nodes * 2 
        self.hinges_arr = self._build_hinges()           # (H=hinges, 2)
        
        self.rest_lengths = self._build_rest_lengths(rest_lengths=rest_lengths)  # rest lengths (float32)
        
    # --- builders ---
    def _build_edges(self) -> jax.Array:
        starts = jnp.arange(self.hinges + 1, dtype=jnp.int32)
        return jnp.stack([starts, starts + 1], axis=1)

    def _build_hinges(self) -> jax.Array:
        if self.hinges <= 0:
            return jnp.empty((0, 2), dtype=jnp.int32)
        starts = jnp.arange(self.hinges, dtype=jnp.int32)
        return jnp.stack([starts, starts + 1], axis=1)

    def _build_rest_lengths(self, rest_lengths: Optional[jax.Array]) -> jax.Array:
        if rest_lengths is not None:
            rl = jnp.asarray(rest_lengths, jnp.float32)
            assert rl.shape == (self.edges,), f"rest_lengths shape {rl.shape} != ({self.edges},)"
            return rl
        return jnp.full((self.edges,), self.L, dtype=jnp.float32)

    def _build_learning_parameters(self, Nin: int, Nout: int) -> None:
        _, _, _, self.DM, self.NE, self.NN, self.output_nodes_arr = learning_funcs.build_incidence(Nin, Nout)
    
    # vectorized helpers (handy + jit-friendly)
    def all_edge_lengths(self, pos: jax.Array) -> jax.Array:
        return jax.vmap(lambda e: self._get_edge_length(pos, e))(jnp.arange(self.edges))

    def all_hinge_angles(self, pos: jax.Array) -> jax.Array:
        return jax.vmap(lambda h: self._get_theta(pos, h))(jnp.arange(self.hinges))
    
    # --- geometry ---      
    def _get_theta(self, pos_arr: jax.Array, hinge: int):
        """Angle at a hinge (radians), CCW positive."""
        fourpoints = pos_arr[self.edges_arr[self.hinges_arr[hinge]]]  # (2,2,2) coords for each edge's endpoints
        # print('pos_arr in _get_theta ', pos_arr)
        # print('edges_arr ', self.edges_arr)
        # print('hinges_arr ', self.hinges_arr)
        # print('fourpoints ', fourpoints)
        vecs = fourpoints[:, 1, :] - fourpoints[:, 0, :]   # (2,2)
        u, v = vecs[:-1], vecs[1:]
        dot = jnp.sum(u * v, axis=-1)
        cross_z = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]  # scalar z of 2D cross
        theta = jnp.arctan2(cross_z, dot)         # signed angle from u -> v        
        return theta
    
    def _get_edge_length(self, pos_arr, edge):
        """Length of one edge given current positions pos: (Npoints,2) float."""
        twopoints = pos_arr[self.edges_arr[edge]]  # (2,)
        vec = twopoints[1, :] - twopoints[0, :]  # (2,)
        return jnp.linalg.norm(vec)
