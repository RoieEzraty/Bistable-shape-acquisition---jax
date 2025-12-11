from __future__ import annotations

import numpy as np
import copy
import jax
import equinox as eqx

from typing import Tuple, List
from numpy import array, zeros
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional
from config import ExperimentConfig

import helpers_builders

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
    
    Numpy side container that stores and updates the evolving geometry from jax instances across training time steps
    Tracks: 1) nodal positions, 2) hinge angles, 3) "buckle state" (upwards or downwards) of each hinge.

    Attributes
    ----------
    pos_arr        - (nodes,2) ndarray, Current nodal positions of the chain.
    pos_arr_in_t   - (nodes,2,T) ndarray, history of nodal positions over the training time.
    theta_arr      - (H,) ndarray, current hinge angles, measured **counter-clockwise (CCW)**.
    theta_arr_in_t - (H,T) ndarray, history of hinge angles over the training time.
    buckle_arr     - (H,S) ndarray, current buckle state of each hinge for each shim.
                     `1` = buckle downwards
                     `-1` = buckle upwards
    buckle_in_t    - (H,S,T) ndarray, history of buckle states over the training time.
    Fx, Fy         - floats, force on tip in x, y directions, 
                     if 2 last nodes or imposed, force is summed over both
    
    tip_torque, tot_torque           - float, current torques:
                                       ``tip_torque`` = local torque acting on the tip, 
                                                        calculated using last 2 nodes and known tip angle
                                       ``tot_torque`` = net torque of the whole chain about the origin.
                                                        calculated using the mean force on tip and total arm angle w.r.t origin
    tip_torque_in_t, tot_torque_in_t - (T,) ndarray, histories of the above torques.
    edge_lengths                     - (edges,) ndarray, current edge lengths, for convenience (last stored snapshot).
    """ 
    # --- instantaneous state ---
    pos_arr: NDArray[np.float32] = eqx.field(static=True)          # (nodes, 2)
    theta_arr: NDArray[np.float32] = eqx.field(static=True)        # (hinges,)
    buckle_arr: NDArray[np.int32] = eqx.field(static=True)         # (hinges, shims)
    Fx: float = eqx.field(static=True)                             # float
    Fy: float = eqx.field(static=True)                             # float
    tip_torque: float = eqx.field(static=True)                     # float, torque just on tip
    tot_torque: float = eqx.field(static=True)                     # float, torque of whole chain

    # --- histories / logs ---
    pos_arr_in_t: NDArray[np.float32] = eqx.field(static=True)     # (nodes, 2, T)
    theta_arr_in_t: NDArray[np.float32] = eqx.field(static=True)   # (hinges, T)
    buckle_in_t: NDArray[np.int32] = eqx.field(static=True)        # (hinges, shims, T)
    Fx_in_t: NDArray[np.float32] = eqx.field(static=True)          # (T,)
    Fy_in_t: NDArray[np.float32] = eqx.field(static=True)          # (T,)
    tip_torque_in_t: NDArray[np.float32] = eqx.field(static=True)  # (T,)
    tot_torque_in_t: NDArray[np.float32] = eqx.field(static=True)  # (T,)

    def __init__(self, Strctr: "StructureClass", Sprvsr: "SupervisorClass", pos_arr: Optional[np.ndarray] = None,
                 buckle_arr: Optional[np.ndarray] = None) -> None:
        """
        Initialize a new state container.

        Parameters
        ----------
        Strctr     - StructureClass. Chain topology (number of nodes/hinges/shims).
        Sprvsr     - SupervisorClass. Provides number of training steps ``T``.
        pos_arr    - (nodes, 2) ndarray, optional. Initial nodal positions. 
                     If None = initialized as a straight chain using `helpers_builders._initiate_pos`.
        buckle_arr - ndarray, optional, shape (hinges, shims). Initial buckle pattern. 
                     If None = initialized with `helpers_builders._initiate_buckle` (all down/up depending on implementation).
        """
        # ------ positions ------
        if pos_arr is None:
            self.pos_arr = helpers_builders._initiate_pos(Strctr.edges+1, Strctr.L).astype(np.float32)
        else:
            self.pos_arr = np.asarray(pos_arr, dtype=np.float32)                              # (nodes, 2) node position
        self.pos_arr_in_t = np.zeros((Strctr.nodes, 2, Sprvsr.T), dtype=np.float32)           # (nodes, 2, T)

        # ------ angles ------
        self.theta_arr = np.zeros((Strctr.hinges,), dtype=np.float32)                         # (H,) hinge angles  
        self.theta_arr_in_t = np.zeros((Strctr.hinges, Sprvsr.T), dtype=np.float32)           # (H,) hinge angles in training time

        # ------ buckle pattern ------
        if buckle_arr is not None:
            self.buckle_arr = buckle_arr                                                    
        else:
            self.buckle_arr = helpers_builders._initiate_buckle(Strctr.hinges, Strctr.shims)  # (H, S) buckle state of shims
        self.buckle_in_t = np.zeros((Strctr.hinges, Strctr.shims, Sprvsr.T))                  # (H, S, T)

        # ------ forces and torques ------
        self.Fx = 0.0
        self.Fx_in_t = np.zeros((Sprvsr.T), dtype=np.float32)                                 # (T,) force on tip in x direction

        self.Fy = 0.0
        self.Fy_in_t = np.zeros((Sprvsr.T), dtype=np.float32)                                 # (T,) force on tip in y direction

        self.tip_torque = 0.0
        self.tip_torque_in_t = np.zeros((Sprvsr.T), dtype=np.float32)                         # (T,) tip torque (not in use?)

        self.tot_torque = 0.0
        self.tot_torque_in_t = np.zeros((Sprvsr.T), dtype=np.float32)                         # (T,) torque from summed tip forces

        # ------ edge lengths (last snapshot) ------
        self.edge_lengths: NDArray[np.float32] = np.zeros((Strctr.edges,), dtype=np.float32)

    # ---------------------------------------------------------------
    # Ingest from EquilibriumClass (JAX → NumPy)
    # ---------------------------------------------------------------
    def _save_data(self, t: int, Strctr: "StructureClass", pos_arr: jax.Array = None, buckle_arr: NDArray = None,
                   Forces: jax.Array = None, control_tip_angle: bool = True) -> None:
        """
        Copy arrays from an EquilibriumClass equilibrium solve (JAX) into this (NumPy) state.

        Parameters
        ----------
        t                 - int. Time-step index (0-based) into the training history.
        Strctr            - StructureClass. Provides geometry (nodes, hinges, etc.).
        pos_arr           - (nodes, 2) jax.Array, optional. Nodal positions from the equilibrium solver.
                            None = a straight chain is initialized.
        buckle_arr        - ndarray, optional. Buckle state from the equilibrium solver.
                            None =  initialized `helpers_builders._initiate_buckle`.
        Forces            - (2 * nodes,) jax.Array, optional. Force vector from the equilibrium solver. 
                            Depending on the `control_tip_angle` flag, sum appropriate components to get net wall forces Fx, Fy.
        control_tip_angle - bool, default True
                            True = tip angle controlled and wall forces come from last *two* nodes. 
                            False = wall forces come from tip node only.
        """
        # ------- positions -------
        if pos_arr is not None:
            self.pos_arr = helpers_builders.jax2numpy(pos_arr)
        else:
            pos_arr = helpers_builders._initiate_pos(Strctr.edges+1, Strctr.L)
            self.pos_arr = helpers_builders.jax2numpy(pos_arr)
        self.pos_arr_in_t[:, :, t] = self.pos_arr

        # ------- buckle state -------
        if buckle_arr is not None:
            self.buckle_arr = helpers_builders.jax2numpy(buckle_arr)
        else:
            self.buckle_arr = helpers_builders.jax2numpy(helpers_builders._initiate_buckle(Strctr.hinges, Strctr.shims))
        self.buckle_in_t[:, :, t] = self.buckle_arr

        # ------- Force normal on wall -------
        if Forces is not None:
            if control_tip_angle:  # tip is controlled, forces are sum over 2 final nodes, each axis on its own
                self.Fx = Forces[-4] + Forces[-2]
                self.Fy = Forces[-3] + Forces[-1]
                # self.Fx = Forces[-4]  # only final node
                # self.Fy = Forces[-3]  # only final node
            else:  # tip is not controlled, forces are on last node
                self.Fx = Forces[-2]
                self.Fy = Forces[-1]
        else:
            Forces = np.array([0, 0, 0, 0])
            self.Fx = 0
            self.Fy = 0
        self.Fx_in_t[t] = self.Fx
        self.Fy_in_t[t] = self.Fy

        # ------- thetas -------
        thetas = Strctr.all_hinge_angles(self.pos_arr)  # (H,) np ndarray
        self.theta_arr = helpers_builders.jax2numpy(thetas).reshape(-1)
        
        # ------- torque -------
        tip_angle = float(helpers_builders._get_tip_angle(self.pos_arr))  # measured from -x
        self.tot_torque = float(helpers_builders.torque(tip_angle, self.Fx, self.Fy))
        self.tot_torque_in_t[t] = self.tot_torque
        self.tip_torque = float(helpers_builders.tip_torque(tip_angle, Forces))
        self.tip_torque_in_t[t] = self.tip_torque

        # ------- edge_lengths -------
        self.edge_lengths = Strctr.all_edge_lengths(self.pos_arr)

    # ---------------------------------------------------------------
    # Buckle update rule
    # ---------------------------------------------------------------
    def buckle(self, Variabs: "VariablesClass", Strctr: "StructureClass", t: int, State_measured: "StateClass") -> bool:
        """
        Update buckle states based on current hinge angles and thresholds (NumPy).

        - If ``buckle = 1`` (e.g. "down") and ``theta < -thresh``, flip to ``-1``.
        - If ``buckle = -1`` (e.g. "up") and ``theta > +thresh``, flip to ``1``.

        Parameters
        ----------
        Variabs        - VariablesClass. Supplies threshold array ``thresh`` of shape (H, S).
        Strctr         - StructureClass. Supplies ``hinges`` and ``shims`` sizes.
        t              - int. Time-step index at which to log.
        State_measured - StateClass. A second state object (typically the "measured") that should mirror the buckle transitions.

        Returns
        -------
        buckle_bool : bool. True if at least one hinge/shim flipped state, False otherwise.
        """
        buckle_bool = False
        buckle_nxt = np.zeros((Strctr.hinges, Strctr.shims), dtype=np.int32)
        for i in range(Strctr.hinges):
            for j in range(Strctr.shims):
                theta_i = self.theta_arr[i]
                thresh_ij = Variabs.thresh[i, j]

                # buckle up (flip 1 -> -1) when angle is too negative
                if self.buckle_arr[i, j] == 1 and theta_i < -thresh_ij:
                    buckle_nxt[i, j] = -1
                    print("buckled up, theta =", theta_i)
                    buckle_bool = True

                # buckle down (flip -1 -> 1) when angle is too positive
                elif self.buckle_arr[i, j] == -1 and theta_i > thresh_ij:
                    buckle_nxt[i, j] = 1
                    print("buckled down, theta =", theta_i)
                    buckle_bool = True

                # no change
                else:
                    buckle_nxt[i, j] = self.buckle_arr[i, j]
        self.buckle_arr = copy.copy(buckle_nxt)
        self.buckle_in_t[:, :, t] = self.buckle_arr

        # buckling is the same also in measurement state
        State_measured.buckle_arr = copy.copy(buckle_nxt)
        State_measured.buckle_in_t[:, :, t] = State_measured.buckle_arr
        return buckle_bool

    # ---------------------------------------------------------------
    # Energy helpers (NumPy-side diagnostics)
    # ---------------------------------------------------------------
    def stretch_energy(self, Variabs: "VariablesClass", Strctr: "StructureClass") -> NDArray[np.float_]:
        """
        stretch energy per edge

        Returns
        -------
        energy : (edges,) ndarray, ``0.5 * k_stretch * (ℓ - ℓ₀)^2`` per edge (up to prefactor).
        """
        lengths = Strctr.all_edge_lengths(self.pos_arr)  # NumPy, (E,)
        rest_lengths = helpers_builders.jax2numpy(Strctr.rest_lengths)  # JAX -> NumPy
        return Variabs.k_stretch * (lengths - rest_lengths) ** 2

    def bending_energy(self, Variabs: "VariablesClass", Strctr: "StructureClass") -> NDArray[np.float_]:
        """
        bending energy per hinge

        Uses the experimental torque curve if available:
        E_bend(θ) ≈ τ(θ) * (θ - buckle * θ_ss)

        Notes
        -----
        - Only meaningful for ``k_type == "Experimental"``, where ``Variabs.torque`` is defined. If torque is None, raise error.
        """
        if Variabs.torque is None:
            raise ValueError("bending_energy requires Variabs.torque (experimental mode).")
        theta_arr = self.theta_arr  # NumPy (H,)
        taus = helpers_builders.jax2numpy(Variabs.torque(theta_arr))  # tau in NumPy 
        effective_thetas_ss = self.buckle_arr * Variabs.thetas_ss
        return taus * (theta_arr - effective_thetas_ss)
