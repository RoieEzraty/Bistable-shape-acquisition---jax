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

import dynamics, helpers_builders

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
    def __init__(self, Variabs: "VariablesClass", Strctr: "StructureClass") -> None:

        self.pos_arr = helpers_builders._initiate_pos(Strctr.hinges)
        self.pos_arr_in_t = np.zeros((Strctr.nodes, 2, Variabs.T))

        self.theta_arr = np.zeros((Strctr.hinges,))    # (H,) hinge angles  
        self.theta_arr_in_t = np.zeros((Strctr.hinges, Variabs.T))    # (H,) hinge angles in training time (usually zeros)  

        self.buckle = np.zeros((Strctr.hinges, Strctr.shims))
        self.buckle_in_t = np.zeros((Strctr.hinges, Strctr.shims, Variabs.T))

    # ---------- ingest from EquilibriumClass ----------

    def _save_data(self, 
                   t: int,
                   Strctr: "StructureClass",
                   pos_arr: jax.Array = None,
                   buckle: jax.Array = None,
                   compute_thetas_if_missing: bool = True) -> None:
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
        # positions (if provided)
        if pos_arr is not None:
            self.pos_arr = helpers_builders.numpify(pos_arr)
        else:
            pos_arr = helpers_builders._initiate_pos(Strctr.hinges)
            self.pos_arr = helpers_builders.numpify(pos_arr)
        self.pos_arr_in_t[:, :, t] = self.pos_arr

        # buckle state
        if buckle is not None:
            self.buckle = helpers_builders.numpify(buckle)
        else:
            self.buckle = helpers_builders.numpify(helpers_builders._initiate_buckle(Strctr.hinges, Strctr.shims))
        self.buckle_in_t[:, :, t] = self.buckle

        # thetas
        if compute_thetas_if_missing:
            thetas = vmap(lambda h: Strctr._get_theta(pos_arr, h))(jnp.arange(Strctr.hinges))
            self.theta_arr = helpers_builders.numpify(thetas).reshape(-1)
            self.theta_arr_in_t[:, t] = self.theta_arr
