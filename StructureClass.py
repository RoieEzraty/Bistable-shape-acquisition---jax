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
from config import ExperimentConfig

import learning_funcs, helpers_builders


# ===================================================
# Class - Structure Variables - arrays hinge connections, edges, etc.
# ===================================================


class StructureClass(eqx.Module):
    """
    Geometric and topological Bistable buckle structure (1D chain in the plane).

    The structure is a chain of straight edges of rest length ``L`` connecting
    point-masses (nodes) in 2D. Neighboring edges meet at hinges, and those
    hinge angles are where the rotational springs live.

    High-level picture
    ------------------
    - Nodes are indexed 0..(nodes-1).
    - Edges are straight segments between consecutive nodes: (0,1), (1,2), ...
    - Hinges live at internal nodes and are defined by the *pair* of edges
      that meet there. For a simple chain:
        * hinge 0  is between edges (0,1)
        * hinge 1  is between edges (1,2)
        * ...
    - Degrees of freedom: each node has (x, y), so there are ``2*nodes`` DOFs.

    This class only knows geometry/topology and which DOFs are fixed. All
    energies and forces are handled elsewhere (EquilibriumClass, VariablesClass).

    Attributes
    ----------
    hinges       - int. Number of hinges (internal joints) along the chain.
    shims        - int. Number of shims per hinge (used by the learning / control layers).
    L            - float. Rest length of each edge. numerical = 1.0 (dimensionless), experimental = 45 mm.
    nodes        - int. Total number of nodes (= hinges + 2).
    edges        - int. Total number of edges (= hinges + 1).
    edges_arr    - ndarray of int, shape (edges, 2). For each edge e, ``edges_arr[e] = (i, j)`` gives the node indices.
    hinges_arr   - ndarray of int, shape (hinges, 2).
                   For each hinge h, ``hinges_arr[h] = (e0, e1)`` gives the edge indices of the two edges that meet at that hinge.
    n_coords     - int. Total number of scalar DOFs = ``2 * nodes``.
    rest_lengths - jax.Array, shape (edges,). Rest length of each edge; by default all equal to ``L``.
    fixed_mask   - jax.Array of bool, shape (2*nodes,). Boolean mask on DOFs: True for DOFs held fixed during all training.
                   node 0 (and optionally node 1) are fixed at the origin.

    DM, NE, NN, output_nodes_arr - optional learning graph objects.
                                   These are populated only if ``update_scheme == "BEASTAL"`` and
                                   DM : incidence matrix connecting inputs (imposed positions) to outputs (measured forces).
                                        shape (NE, NN)
                                   NE : inputs x outputs
                                   NN : input + outputs
                                   output_nodes_arr : indices of outputs
    """   
    # ------ user-provided (static) ------
    hinges: int = eqx.field(static=True)     # number of hinges in the chain
    shims: int = eqx.field(static=True)      # shims per hinge
    L: float = eqx.field(static=True)        # rest length of rods

    # ------ computed in __init__ (static topology/geometry) ------
    nodes: int = eqx.field(init=False, static=True)               # N nodes, (hinges + 2)
    edges: int = eqx.field(init=False, static=True)               # N edges (hinges + 1) 
    edges_arr: jax.Array = eqx.field(init=False, static=True)     # array (hinges+1, 2) int32, nodes to edges
                                                                  # Used inside State.stretch_forces
    hinges_arr: jax.Array = eqx.field(init=False, static=True)    # array (hinges, 2) int32, edges to hinges
    rest_lengths: jax.Array = eqx.field(init=False, static=True)  # (E,) float32   
    n_coords: int = eqx.field(init=False, static=True)            # N coordinates, x and y for each node, so 2*nodes

    # ------ for equilibrium calculation, jax arrays ------
    fixed_mask: jax.Array = eqx.field(init=False, static=True)  # (2*nodes,) bool

    # ------ optional learning graph (only if you call _build_learning_parameters) ------
    DM: Optional[NDArray[int]] = eqx.field(default=None, init=False, static=True)
    NE: Optional[NDArray[int]] = eqx.field(default=None, init=False, static=True)
    NN: Optional[NDArray[int]] = eqx.field(default=None, init=False, static=True)
    output_nodes_arr: Optional[NDArray[int]] = eqx.field(default=None, init=False, static=True)

    def __init__(self, CFG: ExperimentConfig, rest_lengths:  Optional[NDArray[np.float_]] = None,
                 update_scheme: str = 'one_to_one', Nin: Optional[int] = None, Nout: Optional[int] = None,
                 control_first_edge: Optional[bool] = True):
        """
        Build chain geometry from config file.

        Parameters
        ----------
        CFG                - ExperimentConfig. User configuration. uses:
                             CFG.Strctr.H   : number of hinges
                             CFG.Strctr.S   : shims per hinge
                             CFG.Variabs.k_type : "Numerical" or "Experimental"
        rest_lengths       - ndarray, optional. Optional rest lengths for each edge, shape (edges,).
                             If None, all edges have rest length ``L``.
        update_scheme      - {"one_to_one", "BEASTAL"}
                             If "BEASTAL", build the learning incidence structure (DM, NE, NN, ...).
                             If "one_to_one", each force effects only its corresponding position index
        Nin, Nout          - int, optional. N Input / output, 
                             for `learning_funcs.build_incidence` when the "BEASTAL" update scheme is used.
        control_first_edge - bool, default True
                             True = first two nodes are fixed (nodes 0 and 1).
                             False =  only node 0 is fixed. This feeds into `fixed_mask`.
        """
        self.hinges = int(CFG.Strctr.H)
        self.shims = int(CFG.Strctr.S)
        if CFG.Variabs.k_type == 'Experimental':
            self.L = 0.045  # Leon's shims are ~45mm
        else:
            self.L = 1.0

        self.edges_arr = self._build_edges()
        self.edges = int(self.edges_arr.shape[0])
        self.nodes = self.edges + 1
        self.n_coords = self.nodes * 2
        self.hinges_arr = self._build_hinges()
        self.rest_lengths = self._build_rest_lengths(rest_lengths=rest_lengths)
        if update_scheme == 'BEASTAL':
            self.DM, self.NE, self.NN, self.output_nodes_arr = self._build_learning_parameters(Nin, Nout)
        self.fixed_mask = self._build_fixed_mask(control_first_edge)

        # learning fields left as None until _build_learning_parameters is called
       
    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------
    # --- jax builders ---
    def _build_edges(self) -> jax.Array[int]:
        """
        Build the edge list for a simple chain.

        Returns
        -------
        edges_arr - (edges, 2) jax array of int. `edges_arr[e] = (i, j)` with i,j node indices.
        """
        starts = jnp.arange(self.hinges + 1, dtype=int)
        return jnp.stack([starts, starts + 1], axis=1)

    def _build_hinges(self) -> jax.Array[int]:
        """
        Build the hinge-to-edge connectivity.

        Returns
        -------
        hinges_arr : (hinges, 2) jax array of int, `hinges_arr[h] = (e0, e1)` where e0 and e1 are consecutive edges.
        """
        if self.hinges <= 0:
            return jnp.empty((0, 2), dtype=int)
        starts = jnp.arange(self.hinges, dtype=int)
        return jnp.stack([starts, starts + 1], axis=1)

    def _build_rest_lengths(self, rest_lengths: Optional[jax.Array[jnp.float_]]) -> jax.Array[jnp.float_]:
        """
        Build the rest-length array for edges.

        Parameters
        ----------
        rest_lengths : jax array, optional, Explicit rest lengths, shape (edges,). 

        Returns
        -------
        (edges,) jax.Array, Rest length of each edge.
        """
        if rest_lengths is not None:
            rl = jnp.asarray(rest_lengths, jnp.float32)
            assert rl.shape == (self.edges,), f"rest_lengths shape {rl.shape} != ({self.edges},)"
            return rl
        return jnp.full((self.edges,), self.L, dtype=jnp.float32)

    def _build_fixed_mask(self, control_first_edge: bool = True) -> jax.Array:
        """
        Build boolean mask marking fixed DOFs.

        Parameters
        ----------
        control_first_edge - bool, default True
                             True = both node 0 and node 1 are fixed (x and y DOFs).
                             False = only node 0 is fixed.

        Returns
        -------
        fixed_mask - (2*nodes,) jax.Array of bool,  True at fixed DOFs, False elsewhere.
        """
        # ------ fixed and imposed DOFs initialize ------ 
        fixed_mask = jnp.zeros((self.n_coords,), dtype=bool)

        # ------ fixed: node 0, potentially also node 1 ------
        if control_first_edge:
            nodes = [0, 1]  # two nodes are at 0
        else:
            nodes = [0]     # first node at 0, 0

        # ------ set fixed nodes as True in the mask vector ------
        for node in nodes:
            fixed_mask = fixed_mask.at[helpers_builders.dof_idx(node, 0)].set(True)
            fixed_mask = fixed_mask.at[helpers_builders.dof_idx(node, 1)].set(True)
        return fixed_mask

    # --- numpy builders ---
    def _build_learning_parameters(self, Nin: int, Nout: int) -> Tuple[NDArray[np.int_], int, int, NDArray[np.int_]]:
        """
        Build learning-graph incidence matrix and associated sizes.

        Parameters
        ----------
        Nin, Nout - ints. Number of inputs and outputs for the learning graph. 
                    how many positions / forces are accounted for as inputs and how many as outputs

        Returns
        -------
        DM               - (Nin*Nout, Nin+Nout) ndarray of int, Incidence matrix between inputs and outputs.
        NE               - int, Number of edges in the learning graph, which is Nin*Nout.
        NN               - int, Number of nodes in the learning graph, which is Nin+Nout.
        output_nodes_arr - (Nout,) ndarray of int, Indices of output nodes.
        """
        _, _, _, DM, NE, NN, output_nodes_arr = learning_funcs.build_incidence(Nin, Nout)
        return DM, NE, NN, output_nodes_arr

    # ------------------------------------------------------------------
    # JAX-based geometry (used in EquilibriumClass)
    # ------------------------------------------------------------------  
    @staticmethod
    def _normalize(v: jax.Array, eps: float = 1e-9) -> jax.Array:
        """Normalize a 2D vector with a small epsilon for numerical safety."""
        n = jnp.linalg.norm(v)
        n_safe = jnp.maximum(n, eps)
        return v / n_safe      

    def _angle_from_uv(self, u: jax.Array, v: jax.Array) -> jax.Array:
        """
        Signed angle between two 2D vectors u and v (JAX).

        Returns
        -------
        angle - jax.Array (scalar). Angle in radians, positive for counter-clockwise rotation from u to v.
        """
        un = self._normalize(u)
        vn = self._normalize(v)
        dot = jnp.clip(jnp.dot(un, vn), -1.0, 1.0)  # not strictly needed for atan2, but helps near ±1
        cross = un[0] * vn[1] - un[1] * vn[0]
        return jnp.arctan2(cross, dot)

    def _get_theta(self, pos_arr: jax.Array, hinge: int) -> jax.Array:
        """Angle at a hinge (radians), CCW positive.

        Returns
        -------
        angle - jax.Array (scalar). Hinge angle in radians.
        """
        edges = self.edges_arr
        hinges = self.hinges_arr
        e0, e1 = hinges[hinge]
        i0, i1 = edges[e0]
        j0, j1 = edges[e1]
        u = pos_arr[i1] - pos_arr[i0]
        v = pos_arr[j1] - pos_arr[j0]
        return self._angle_from_uv(u, v)

    def _get_edge_length(self, pos_arr: jax.Array, edge: int) -> jax.Array:
        """
        Length of one edge given current positions pos: (Npoints,2) float.
        
        Returns
        -------
        length - jax.Array (scalar). Euclidean length of the edge.
        """
        edges = self.edges_arr
        i, j = edges[edge]
        return jnp.linalg.norm(pos_arr[j] - pos_arr[i])

    # ------------------------------------------------------------------
    # Numpy-based geometry (for convenience / plotting)
    # ------------------------------------------------------------------
    def all_edge_lengths(self, pos_arr: NDArray[np.float_]) -> NDArray[np.float_]:
        """
        Compute the length of all edges for a given configuration.

        Parameters
        ----------
        pos_arr - (nodes, 2) ndarray of node positions.

        Returns
        -------
        lengths - (edges,) ndarray of Euclidean lengths of each edge. All should be ~Strctr.L
        """
        vecs = pos_arr[self.edges_arr[:, 1]] - pos_arr[self.edges_arr[:, 0]]
        return np.linalg.norm(vecs, axis=1)

    def all_hinge_angles(self, pos_arr: NDArray[np.float_]) -> NDArray[np.float_]:
        """
        Compute all hinge angles (numpy version). Between two incident edges, positive is counter-clockwise rotation.

        Parameters
        ----------
        pos_arr - (nodes, 2) ndarray of float of node positions.

        Returns
        -------
        angles - (hinges,) ndarray of float32, hinge angles in radians.
        """
        edge_nodes = self.edges_arr[self.hinges_arr]             # (H, 2, 2)
        pts = pos_arr[edge_nodes]                                # (H, 2, 2, 2)

        vecs = pts[..., 1, :] - pts[..., 0, :]                   # (H, 2, 2)
        u, v = vecs[:, 0, :], vecs[:, 1, :]                      # (H, 2), (H, 2)

        dot = np.sum(u * v, axis=-1)                             # (H,)
        cross_z = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]  # (H,)
        return np.arctan2(cross_z, dot).astype(np.float32)
