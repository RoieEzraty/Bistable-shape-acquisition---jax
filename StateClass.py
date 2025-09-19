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


# ===================================================
# Class - State Variables - node positions, forcing, etc.
# ===================================================


class StateClass:
    """
    Dynamic state of the chain (positions + hinge stiffness regime).
    """   
    def __init__(self, Variabs: "VariablesClass", Strctr: "StructureClass", T: int) -> None:
        
        self.pos_arr = jnp.array([Strctr.edges, ])  # (H, T) hinge angles at rest (usually zeros) 
        self.theta_arr = jnp.array([Strctr.hinges, ])
        self.buckle = jnp.array([Strctr.hinges, Strctr.shims])
        self.pos_arr_in_t = jnp.array([Strctr.edges, T])
        self.theta_arr_in_t = jnp.array([Strctr.hinges, T])  # (H, T) hinge angles at rest (usually zeros)  
        self.buckle_in_t = jnp.array([Strctr.hinges, Strctr.shims, T])

    # ---------- helpers ----------

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

     # ---------- ingest from EquilibriumClass ----------

    def _save_data(self, Strctr: "StructureClass", compute_thetas_if_missing: bool = True) -> None:
        """
        Copy arrays from an EquilibriumClass instance into this StateClass.
        Expected attributes on Eq:
          - final_pos : (N,2)
          - Eq.traj_pos  : (T,N,2)
          - Eq.traj_vel  : (T,N,2)  [optional, not stored here but available if you want]
          - optionally Eq.traj_thetas : (T,H)
          - optionally Eq.final_thetas: (H,)
          - optionally Eq.buckle, Eq.buckle_in_t

        If thetas are missing and compute_thetas_if_missing=True, they are computed from traj_pos.
        """
        # positions
        if hasattr(Eq, "final_pos"):
            self.pos_arr = jnp.asarray(Eq.final_pos, dtype=jnp.float32)
        if hasattr(Eq, "traj_pos"):
            self.pos_arr_in_t = jnp.asarray(Eq.traj_pos, dtype=jnp.float32)

        # buckle (if provided by Eq); otherwise keep defaults
        if hasattr(Eq, "buckle"):
            self.buckle = jnp.asarray(Eq.buckle, dtype=jnp.int32)
            # update time-broadcast
            self.buckle_in_t = jnp.broadcast_to(self.buckle, (self.T, self.H, self.S))
        if hasattr(Eq, "buckle_in_t"):
            self.buckle_in_t = jnp.asarray(Eq.buckle_in_t, dtype=jnp.int32)

        # thetas
        if hasattr(Eq, "final_thetas"):
            self.theta_arr = jnp.asarray(Eq.final_thetas, dtype=jnp.float32)
        if hasattr(Eq, "traj_thetas"):
            self.theta_arr_in_t = jnp.asarray(Eq.traj_thetas, dtype=jnp.float32)
        elif compute_thetas_if_missing and hasattr(Eq, "traj_pos"):
            # compute from geometry if not present
            self.theta_arr_in_t = self._compute_thetas_over_traj(Strctr, self.pos_arr_in_t)
            # also fill the snapshot from the last frame
            self.theta_arr = self.theta_arr_in_t[-1]

    # ---------- convenience: numpy views for plotting ----------

    def np_pos(self) -> NDArray[np.float_]:
        return np.asarray(self.pos_arr)

    def np_traj_pos(self) -> NDArray[np.float_]:
        return np.asarray(self.pos_arr_in_t)

    def np_thetas(self) -> NDArray[np.float_]:
        return np.asarray(self.theta_arr)

    def np_traj_thetas(self) -> NDArray[np.float_]:
        return np.asarray(self.theta_arr_in_t)
