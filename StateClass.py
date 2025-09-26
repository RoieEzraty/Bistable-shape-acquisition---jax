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
    from SupervisorClass import SupervisorClass


# ===================================================
# Class - State Variables - node positions, forcing, etc.
# ===================================================


class StateClass:
    """
    Dynamic state of the chain (positions + hinge stiffness regime).
    
    Stores and updates the evolving geometry and hinge regimes 
    of the chain across training time steps. Tracks nodal positions, hinge angles, 
    and the "buckle state" (upwards or downwards) of each hinge.

    Attributes
    ----------
    pos_arr : ndarray, shape (N,2)
        Current nodal positions of the chain (N = number of nodes).
    pos_arr_in_t : ndarray, shape (N,2,T)
        History of nodal positions over the training time.
    theta_arr : ndarray, shape (H,)
        Current hinge angles, measured **counter-clockwise (CCW)**.
    theta_arr_in_t : ndarray, shape (H,T)
        History of hinge angles over the training time.
    buckle_arr : ndarray, shape (H,S)
        Current buckle state of each hinge for each shim.
        - `1`  → buckle downwards
        - `-1` → buckle upwards
    buckle_in_t : ndarray, shape (H,S,T)
        History of buckle states over the training time.
    Fx, Fy: floats, force on tip in x, y directions
    tip_torque: float, torque on tip

    Methods
    -------
    _save_data(t, Strctr, pos_arr=None, buckle_arr=None, compute_thetas_if_missing=True)
        Copy arrays from an `EquilibriumClass` instance (or raw data) 
        into this state. Updates positions, buckle states, and hinge angles.
    position_tip(Sprvsr, t)
        Store the current supervised tip position at time `t` from training dataset.
    buckle(Variabs, Strctr, t)
        Update hinge buckle states based on current hinge angles 
        and threshold values. Buckle transitions occur when:
          - buckle = 1 (downwards) flips to -1 if angle < -threshold (CCW).
          - buckle = -1 (upwards) flips to 1 if angle > +threshold (CCW).
    """ 
    def __init__(self, Variabs: "VariablesClass", Strctr: "StructureClass", Sprvsr: "SupervisorClass",
                 pos_arr: np.array = None, buckle_arr: np.array = None) -> None:

        if pos_arr is not None:
            self.pos_arr = pos_arr
        else:
            self.pos_arr = helpers_builders._initiate_pos(Strctr.hinges)
        self.pos_arr_in_t = np.zeros((Strctr.nodes, 2, Sprvsr.T))

        if isinstance(Sprvsr.tip_angle_update_in_t, np.ndarray):
            self.tip_angle_in_t = np.zeros((Sprvsr.T,))

        self.theta_arr = np.zeros((Strctr.hinges,))    # (H,) hinge angles  
        self.theta_arr_in_t = np.zeros((Strctr.hinges, Sprvsr.T))    # (H,) hinge angles in training time (usually zeros)  

        if buckle_arr is not None:
            self.buckle_arr = buckle_arr
        else:
            self.buckle_arr = helpers_builders._initiate_buckle(Strctr.hinges, Strctr.shims)
        self.buckle_in_t = np.zeros((Strctr.hinges, Strctr.shims, Sprvsr.T))

        self.Fx = 0.0
        self.Fx_in_t = np.zeros(Sprvsr.T)

        self.Fy = 0.0
        self.Fy_in_t = np.zeros(Sprvsr.T)

        self.tip_torque = 0.0
        self.tip_torque_in_t = np.zeros(Sprvsr.T)

    # ---------- ingest from EquilibriumClass ----------

    def _save_data(self, 
                   t: int,
                   Strctr: "StructureClass",
                   pos_arr: jax.Array = None,
                   buckle_arr: jax.Array = None,
                   Forces: jax.Array = None,
                   compute_thetas_if_missing: bool = True) -> None:
        """
        Copy arrays from an EquilibriumClass instance into this StateClass.
        Expected attributes on Eq:
          - final_pos : (N,2)
          - ???

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
        if buckle_arr is not None:
            self.buckle_arr = helpers_builders.numpify(buckle_arr)
        else:
            self.buckle_arr = helpers_builders.numpify(helpers_builders._initiate_buckle(Strctr.hinges, Strctr.shims))
        self.buckle_in_t[:, :, t] = self.buckle_arr

        # Force normal on wall
        if Forces is not None:
            self.Fx = helpers_builders.numpify(Forces)[-1][-2]
            self.Fy = helpers_builders.numpify(Forces)[-1][-1]
        else:
            self.Fx = 0
            self.Fy = 0
        self.Fx_in_t[t] = self.Fx
        self.Fy_in_t[t] = self.Fy

        # thetas
        if compute_thetas_if_missing:
            thetas = vmap(lambda h: Strctr._get_theta(pos_arr, h))(jnp.arange(Strctr.hinges))
            self.theta_arr = helpers_builders.numpify(thetas).reshape(-1)
            self.theta_arr_in_t[:, t] = self.theta_arr
        # tip angle measured from -x
        self.tip_angle = helpers_builders._get_tip_angle(self.pos_arr)

        self.tip_torque = helpers_builders.torque(self.tip_angle, self.Fx, self.Fy)
        self.tip_torque_in_t[t] = self.tip_torque

    def position_tip(self, Sprvsr: "SupervisorClass", t: int, modality: str = "measurement") -> None:
        if modality == "measurement":
            self.tip_pos = Sprvsr.tip_pos_in_t[t]
        elif modality == "update":
            self.tip_pos = Sprvsr.tip_pos_update_in_t[t]

        if isinstance(Sprvsr.tip_angle_update_in_t, np.ndarray):
            if modality == "measurement":
                self.tip_angle = Sprvsr.tip_angle_in_t[t]
            elif modality == "update":
                self.tip_angle = Sprvsr.tip_angle_update_in_t[t]

    def buckle(self, Variabs: "VariablesClass", Strctr: "StructureClass", t, State_measured: "StateClass"):
        buckle_nxt = np.zeros((Strctr.hinges, Strctr.shims))
        for i in range(Strctr.hinges):
            for j in range(Strctr.shims):
                if self.buckle_arr[i, j] == 1 and self.theta_arr[i] < -Variabs.thresh[i, j]:  # buckle up since thetas are CCwise
                    buckle_nxt[i, j] = -1
                elif self.buckle_arr[i, j] == -1 and self.theta_arr[i] > Variabs.thresh[i, j]:  # buckle down, thetas are CCwise
                    buckle_nxt[i, j] = 1
                else:
                    buckle_nxt[i, j] = self.buckle_arr[i, j]
        self.buckle_arr = copy.copy(buckle_nxt)
        self.buckle_in_t[:, :, t] = self.buckle_arr

        # buckling is the same also in measurement state
        State_measured.buckle_arr = copy.copy(buckle_nxt)
        State_measured.buckle_in_t
