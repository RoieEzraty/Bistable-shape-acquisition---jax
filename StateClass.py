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
    buckle(Variabs, Strctr, t)
        Update hinge buckle states based on current hinge angles 
        and threshold values. Buckle transitions occur when:
          - buckle = 1 (downwards) flips to -1 if angle < -threshold (CCW).
          - buckle = -1 (upwards) flips to 1 if angle > +threshold (CCW).
    """ 

    # --- instantaneous state ---
    pos_arr: NDArray[np.float32] = eqx.field(static=True)          # (nodes, 2)
    theta_arr: NDArray[np.float32] = eqx.field(static=True)        # (hinges,)
    buckle_arr: NDArray[np.int32] = eqx.field(static=True)         # (hinges, shims)
    Fx: float = eqx.field(static=True)
    Fy: float = eqx.field(static=True)
    tip_torque: float = eqx.field(static=True)

    # --- histories / logs ---
    pos_arr_in_t: NDArray[np.float32] = eqx.field(static=True)     # (nodes, 2, T)
    theta_arr_in_t: NDArray[np.float32] = eqx.field(static=True)   # (hinges, T)
    buckle_in_t: NDArray[np.int32] = eqx.field(static=True)        # (hinges, shims, T)
    Fx_in_t: NDArray[np.float32] = eqx.field(static=True)          # (T,)
    Fy_in_t: NDArray[np.float32] = eqx.field(static=True)          # (T,)
    tip_torque_in_t: NDArray[np.float32] = eqx.field(static=True)  # (T,)

    def __init__(self, Variabs: "VariablesClass", Strctr: "StructureClass", Sprvsr: "SupervisorClass",
                 pos_arr: Optional[np.ndarray] = None, buckle_arr: Optional[np.ndarray] = None) -> None:

        # current state
        if pos_arr is None:
            self.pos_arr = helpers_builders._initiate_pos(Strctr.edges+1, Strctr.L).astype(np.float32)
        else:
            self.pos_arr = np.asarray(pos_arr, dtype=np.float32)
        self.pos_arr_in_t = np.zeros((Strctr.nodes, 2, Sprvsr.T), dtype=np.float32)

        self.theta_arr = np.zeros((Strctr.hinges,), dtype=np.float32)    # (H,) hinge angles  
        self.theta_arr_in_t = np.zeros((Strctr.hinges, Sprvsr.T), dtype=np.float32)    # (H,) hinge angles in training time

        if buckle_arr is not None:
            self.buckle_arr = buckle_arr
        else:
            self.buckle_arr = helpers_builders._initiate_buckle(Strctr.hinges, Strctr.shims)
        self.buckle_in_t = np.zeros((Strctr.hinges, Strctr.shims, Sprvsr.T))

        self.Fx = 0.0
        self.Fx_in_t = np.zeros((Sprvsr.T), dtype=np.float32)

        self.Fy = 0.0
        self.Fy_in_t = np.zeros((Sprvsr.T), dtype=np.float32)

        self.tip_torque = 0.0
        self.tip_torque_in_t = np.zeros((Sprvsr.T), dtype=np.float32)

    # ---------- ingest from EquilibriumClass ----------

    def _save_data(self, 
                   t: int,
                   Strctr: "StructureClass",
                   pos_arr: jax.Array = None,
                   buckle_arr: NDArray = None,
                   Forces: jax.Array = None,
                   compute_thetas_if_missing: bool = True,
                   control_tip_angle: bool = True) -> None:
        """
        Copy arrays from an EquilibriumClass equilibrium solve (JAX) into this (NumPy) state.

        If thetas are missing and compute_thetas_if_missing=True, they are computed from traj_pos.
        """
        # positions (if provided)
        if pos_arr is not None:
            self.pos_arr = helpers_builders.numpify(pos_arr)
        else:
            pos_arr = helpers_builders._initiate_pos(Strctr.edges+1, Strctr.L)
            self.pos_arr = helpers_builders.numpify(pos_arr)
        self.pos_arr_in_t[:, :, t] = self.pos_arr

        # buckle state
        if buckle_arr is not None:
            self.buckle_arr = helpers_builders.numpify(buckle_arr)
        else:
            self.buckle_arr = helpers_builders.numpify(helpers_builders._initiate_buckle(Strctr.hinges, Strctr.shims))
        self.buckle_in_t[:, :, t] = self.buckle_arr

        # Force normal on wall taken from the last row if provided
        if Forces is not None:
            if control_tip_angle:  # tip is controlled, forces are on one before last node
                self.Fx = helpers_builders.numpify(Forces)[-1][-4] + helpers_builders.numpify(Forces)[-1][-2]
                self.Fy = helpers_builders.numpify(Forces)[-1][-3] + helpers_builders.numpify(Forces)[-1][-1]
            else:  # tip is not controlled, forces are on last node
                self.Fx = helpers_builders.numpify(Forces)[-1][-2]
                self.Fy = helpers_builders.numpify(Forces)[-1][-1]
        else:
            self.Fx = 0
            self.Fy = 0
        self.Fx_in_t[t] = self.Fx
        self.Fy_in_t[t] = self.Fy

        # thetas - hinge angles (JAX compute -> NumPy store)
        if compute_thetas_if_missing:
            # thetas = vmap(lambda h: Strctr._get_theta(pos_arr, h))(jnp.arange(Strctr.hinges))
            thetas = Strctr.all_hinge_angles(self.pos_arr)  # (H,)
            # self.theta_arr = np.asarray(jax.device_get(thetas), dtype=np.float32).reshape(-1)
            self.theta_arr = helpers_builders.numpify(thetas).reshape(-1)
        
        # tip angle measured from -x
        tip_angle = float(helpers_builders._get_tip_angle(self.pos_arr))
        self.tip_torque = float(helpers_builders.torque(tip_angle, self.Fx, self.Fy))
        self.tip_torque_in_t[t] = self.tip_torque

        self.edge_lengths = Strctr.all_edge_lengths(self.pos_arr)

    def buckle(self, Variabs: "VariablesClass", Strctr: "StructureClass", t: int, State_measured: "StateClass"):
        """Update buckle states based on current hinge angles and thresholds (NumPy)."""
        buckle_nxt = np.zeros((Strctr.hinges, Strctr.shims), dtype=np.int32)
        for i in range(Strctr.hinges):
            for j in range(Strctr.shims):
                if self.buckle_arr[i, j] == 1 and self.theta_arr[i] < -Variabs.thresh[i, j]:  # buckle up, thetas are CCwise
                    buckle_nxt[i, j] = -1
                    print('buckled up, theta=', self.theta_arr[i])
                elif self.buckle_arr[i, j] == -1 and self.theta_arr[i] > Variabs.thresh[i, j]:  # buckle down, thetas are CCwise
                    buckle_nxt[i, j] = 1
                    print('buckled down, theta=', self.theta_arr[i])
                else:
                    buckle_nxt[i, j] = self.buckle_arr[i, j]
        self.buckle_arr = copy.copy(buckle_nxt)
        self.buckle_in_t[:, :, t] = self.buckle_arr

        # buckling is the same also in measurement state
        State_measured.buckle_arr = copy.copy(buckle_nxt)
        State_measured.buckle_in_t[:, :, t] = State_measured.buckle_arr

    def stretch_energy(self, Variabs: "VariablesClass", Strctr: "StructureClass") -> NDArray:
        """
        stretch energy per edge
        """
        return Variabs.k_stretch*(Strctr.all_edge_lengths(self.pos_arr) - Strctr.rest_lengths)**2

    def bending_energy(self, Variabs: "VariablesClass", Strctr: "StructureClass") -> NDArray:
        """
        bending energy per hinge
        """
        return Variabs.torque(self.theta_arr)*(self.theta_arr - self.buckle_arr * Variabs.thetas_ss)

