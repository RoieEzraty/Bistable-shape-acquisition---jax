from __future__ import annotations

import numpy as np
import copy
import jax
import jax.numpy as jnp
import equinox as eqx
from jax import vmap

from typing import Tuple, List
from numpy import array, zeros
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional

import helpers_builders, learning_funcs

if TYPE_CHECKING:
    from StructureClass import StructureClass
    from StateClass import StateClass
    from EquilibriumClass import EquilibriumClass
    from VariablesClass import VariablesClass


# ===================================================
# Class - Supervisor Variables - training set, losses, etc.
# ===================================================


class SupervisorClass:
    """
    Variables that are by the external supervisor in the experiment
    """  
    # --- configuration / hyperparams ---
    T: int = eqx.field(static=True)                         # training set length
    alpha: float = eqx.field(static=True)                   # step size
    update_scheme: str = eqx.field(static=True)             # "one_to_one" | "BEASTAL"
    control_tip_angle: bool = eqx.field(static=True)

    # --- desired targets (fixed-size buffers; NumPy, mutable at runtime) ---
    desired_buckle_arr: NDArray[np.int32] = eqx.field(static=True)
    desired_pos_in_t: NDArray[np.float32] = eqx.field(init=False, static=True)     # (nodes, 2, T)
    desired_Fx_in_t: NDArray[np.float32] = eqx.field(init=False, static=True)      # (T,)
    desired_Fy_in_t: NDArray[np.float32] = eqx.field(init=False, static=True)      # (T,)
    desired_tau_in_t: Optional[NDArray[np.float32]] = eqx.field(default=None, init=False, static=True)  # (T,)

    # --- dataset inputs (what tip we command at each step) ---
    tip_pos_in_t: NDArray[np.float32] = eqx.field(init=False, static=True)         # (T, 2)
    tip_angle_in_t: Optional[NDArray[np.float32]] = eqx.field(default=None, init=False, static=True)    # (T,)

    # --- running logs / losses ---
    loss_in_t: NDArray[np.float32] = eqx.field(init=False, static=True)            # (T, 2) or (T, 3)
    tip_pos_update_in_t: NDArray[np.float32] = eqx.field(init=False, static=True)  # (T, 2)
    tip_angle_update_in_t: Optional[NDArray[np.float32]] = eqx.field(default=None, init=False, static=True)  # (T,)

    # --- scratch (most recent loss vector) ---
    loss: NDArray[np.float32] = eqx.field(init=False, static=True)                 # (2,) or (3,) 

    def __init__(self, Strctr, alpha: float, T: int, desired_buckle_arr: np.ndarray, sampling='Uniform',
                 control_tip_angle: bool = True, update_scheme: str = 'one_to_one') -> None:
        
        self.T = int(T)  # total training set size (and algorithm time, not to confuse with time to reach equilibrium state)
        self.alpha = float(alpha)
        self.update_scheme = str(update_scheme)
        self.control_tip_angle = bool(control_tip_angle)

        # Desired/targets
        self.desired_buckle_arr = np.asarray(desired_buckle_arr, dtype=np.int32)
        self.desired_pos_in_t = np.zeros((Strctr.nodes, 2, T), dtype=np.float32)
        self.desired_Fx_in_t = np.zeros((T), dtype=np.float32)
        self.desired_Fy_in_t = np.zeros((T), dtype=np.float32)
        if control_tip_angle:
            self.desired_tau_in_t = np.zeros((T), dtype=np.float32)

        # Dataset (commands)
        self.tip_pos_in_t = np.zeros((self.T, 2), dtype=np.float32)
        if self.control_tip_angle:
            self.tip_angle_in_t = np.zeros((self.T,), dtype=np.float32)

        # Logs / updates
        self.loss_in_t = np.zeros((self.T, 3 if self.control_tip_angle else 2), dtype=np.float32)
        self.tip_pos_update_in_t = np.zeros((self.T, 2), dtype=np.float32)
        if self.control_tip_angle:
            self.tip_angle_update_in_t = np.zeros((self.T,), dtype=np.float32)

        # Last loss vector (shape matches control mode)
        self.loss = np.zeros((3 if self.control_tip_angle else 2,), dtype=np.float32)

    def create_dataset(self, Strctr: "StructureClass", sampling: str) -> None:
        if sampling == 'uniform':
            x_pos_in_t = np.random.uniform((Strctr.edges-1)*Strctr.L, Strctr.edges*Strctr.L, size=self.T)
            y_pos_in_t = np.random.uniform(-Strctr.L/3, Strctr.L/3, size=self.T)
            self.tip_pos_in_t = np.stack(((x_pos_in_t), (y_pos_in_t.T)), axis=1)
            if self.control_tip_angle and self.tip_angle_in_t is not None:
                self.tip_angle_in_t[:] = np.random.uniform(-np.pi / 5, np.pi / 5, size=self.T).astype(np.float32)
        elif sampling == 'flat':
            end = float(Strctr.hinges + 2)
            tip_pos = np.array([end, 0], dtype=np.float32)
            self.tip_pos_in_t[:] = np.tile(tip_pos, (self.T, 1))
            if self.control_tip_angle and self.tip_angle_in_t is not None:
                self.tip_angle_in_t[:] = 0.0
        elif sampling == 'almost flat':
            end = float(Strctr.hinges + 2)
            tip_pos = np.array([end,  0.0], dtype=np.float32)  # flat arrangement

            # tiny noise around each position (tune scale as you like)
            noise_scale = 0.2 * Strctr.L
            noise = noise_scale * np.random.randn(self.T, 2).astype(np.float32)

            self.tip_pos_in_t[:] = tip_pos + noise

            if self.control_tip_angle and self.tip_angle_in_t is not None:
                self.tip_angle_in_t[:] = 0.0
        else:
            raise ValueError(f"Incompatible sampling='{sampling}'")

    def set_desired(self, pos_arr: jax.Array, Fx: float, Fy: float, t: int, tau: Optional[float] = None) -> None:
        """Store ground-truth targets for step t."""
        self.desired_pos_in_t[:, :, t] = helpers_builders.numpify(pos_arr)
        self.desired_Fx_in_t[t] = float(Fx)
        self.desired_Fy_in_t[t] = float(Fy)
        if self.control_tip_angle and self.desired_tau_in_t is not None and tau is not None:
            self.desired_tau_in_t[t] = float(tau)

    def calc_loss(self, t: int, Fx: float, Fy: float, tau: Optional[float] = None) -> None:
        """Compute loss vector (Fx,Fy[,tau]) at step t and log it."""
        if self.control_tip_angle and tau is not None and self.desired_tau_in_t is not None:
            self.loss = np.array(
                [self.desired_Fx_in_t[t] - Fx, self.desired_Fy_in_t[t] - Fy, self.desired_tau_in_t[t] - tau],
                dtype=np.float32,
            )
        else:
            self.loss = np.array(
                [self.desired_Fx_in_t[t] - Fx, self.desired_Fy_in_t[t] - Fy],
                dtype=np.float32,
            )
        self.loss_in_t[t, : self.loss.shape[0]] = self.loss

    def calc_update_tip(self, t: int, Strctr: "StructureClass", Variabs: "VariablesClass", State: "StateClass",
                        prev_tip_update_pos: Optional[np.ndarray] = None,
                        current_tip_angle: Optional[float] = None,
                        prev_tip_update_angle: Optional[float] = None,) -> None:
        """Compute next tip position/angle commands from current loss and state (pure NumPy)."""
        # Normalised inputs/outputs (NumPy)
        inputs_normalized = np.array([State.tip_pos[0]/Variabs.norm_pos, State.tip_pos[1]/Variabs.norm_pos,
                                      State.tip_angle/Variabs.norm_angle], dtype=np.float32)
        outputs_normalized = np.array([State.Fx/Variabs.norm_force, State.Fy/Variabs.norm_force,
                                       State.tip_torque/Variabs.norm_torque], dtype=np.float32)

        # --- BEASTAL or one_to_one ---
        if self.update_scheme == 'BEASTAL':
            grad_loss_vec = learning_funcs.grad_loss_FC(Strctr.NE, inputs_normalized, outputs_normalized, Strctr.DM,
                                                        Strctr.output_nodes_arr, self.loss)

            update_vec = - self.alpha * np.matmul(Strctr.DM_dagger, grad_loss_vec)
            delta_tip = update_vec[0:2]
            delta_angle = update_vec[2] if self.control_tip_angle else 0.0
        elif self.update_scheme == 'one_to_one':
            delta_tip = - self.alpha * self.loss[:2]
            delta_angle = + self.alpha * self.loss[2] if (self.control_tip_angle and self.loss.size == 3) else 0.0
        else:
            raise ValueError(f"Unknown update_scheme='{self.update_scheme}'")

        # insert into tip_pos_update
        if prev_tip_update_pos is None:
            prev_tip_update_pos = self.tip_pos_update_in_t[t-1, :]
        # delta_tip = self.alpha*(np.array([Fx, Fy]) - current_tip_pos)*(self.loss) * ([2, 0.5])  # old, not BEASTAL
        self.tip_pos_update_in_t[t, :] = prev_tip_update_pos + delta_tip

        # Angle update (only if enabled)
        if self.control_tip_angle and self.tip_angle_update_in_t is not None:
            if prev_tip_update_angle is None and t > 0:
                prev_tip_update_angle = float(self.tip_angle_update_in_t[t - 1])
            elif prev_tip_update_angle is None:
                prev_tip_update_angle = 0.0
            self.tip_angle_update_in_t[t] = float(prev_tip_update_angle + float(delta_angle))
