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

import helpers_builders, learning_funcs

if TYPE_CHECKING:
    from StructureClass import StructureClass
    from EquilibriumClass import EquilibriumClass
    from VariablesClass import VariablesClass


# ===================================================
# Class - Supervisor Variables - training set, losses, etc.
# ===================================================


class SupervisorClass:
    """
    Variables that are by the external supervisor in the experiment
    """   
    def __init__(self, Strctr, alpha: float, T: int, desired_buckle_arr: jax.Array, sampling='Uniform',
                 control_tip_angle=True) -> None:
        
        self.T = T  # total training set size (and algorithm time, not to confuse with time to reach equilibrium state)
        self.alpha = alpha
        self.desired_buckle_arr = desired_buckle_arr
        self.desired_pos_in_t = np.zeros((Strctr.nodes, 2, T))
        self.desired_Fx_in_t = np.zeros(T)
        self.desired_Fy_in_t = np.zeros(T)
        self.loss_in_t = np.zeros((T, 2))
        self.tip_pos_update_in_t = np.zeros((T, 2))
        if control_tip_angle:  # if controlling also the tip angle
            self.tip_angle_update_in_t = np.zeros(T)

    def create_dataset(self, Strctr: "StructureClass", sampling: str) -> None:
        if sampling == 'Uniform':
            x_pos_in_t = np.random.uniform((Strctr.edges-1)*Strctr.L, Strctr.edges*Strctr.L, size=self.T)
            y_pos_in_t = np.random.uniform(-Strctr.L/3, Strctr.L/3, size=self.T)
            self.tip_pos_in_t = np.stack(((x_pos_in_t), (y_pos_in_t.T)), axis=1)
            if isinstance(self.tip_angle_update_in_t, np.ndarray):  # if controlling also the tip angle
                self.tip_angle_in_t = np.random.uniform(-np.pi/5, np.pi/5, size=self.T)
        elif sampling == 'Flat':
            end = Strctr.hinges + 2
            tip_pos = np.array([end, 0])
            self.tip_pos_in_t = np.tile(tip_pos, (self.T, 1))
            if isinstance(self.tip_angle_update_in_t, np.ndarray):  # if controlling also the tip angle
                self.tip_angle_in_t = np.zeros(self.T)
        else:
            print('User specified incompatible sampling')

    def set_desired(self, pos_arr: jax.Array, Fx: float, Fy: float, t: int) -> None:
        self.desired_pos_in_t[:, :, t] = pos_arr
        self.desired_Fx_in_t[t] = Fx
        self.desired_Fy_in_t[t] = Fy

    def calc_loss(self, Fx: float, Fy: float, t) -> None:
        self.loss = np.array([self.desired_Fx_in_t[t] - Fx, self.desired_Fy_in_t[t] - Fy])
        self.loss_in_t[t, :] = self.loss

    def calc_update_tip(self, t: int, State: "StateClass", Strctr: "StructureClass", prev_tip_update_pos: np.array = None,
                        current_tip_angle: np.float = None, prev_tip_update_angle: np.float = None) -> None:

        # torque = np.cos(current_tip_angle)*Fy-np.sin(current_tip_angle)*Fx
        norm_pos = Strctr.hinges*Strctr.L
        norm_angle = np.pi
        print('Roie dont forger to normalize force')
        # norm_force = ?
        # norm_torque = ?
        inputs_normalized = np.array([State.tip_pos[0]/norm_pos, State.tip_pos[1]/norm_pos, State.tip_angle/norm_angle])
        outputs_normalized = np.array([State.Fx/norm_force, State.Fy/norm_force, State.tip_torque/norm_torque])

        # --- BEASTAL ---
        grad_loss_vec = learning_funcs.grad_loss_FC(Strctr.NE, inputs_normalized, outputs_normalized, Strctr.DM,
                                                    Strctr.output_nodes_arr, self.loss)

        update_vec = - self.alpha * np.matmul(Strctr.DM_dagger, grad_loss_vec)
        delta_tip = update_vec[0:2]
        delta_angle = update_vec[2]

        # insert into tip_pos_update
        if prev_tip_update_pos is None:
            prev_tip_update_pos = self.tip_pos_update_in_t[t-1, :]
        # delta_tip = self.alpha*(np.array([Fx, Fy]) - current_tip_pos)*(self.loss) * ([2, 0.5])  # old, not BEASTAL
        self.tip_pos_update_in_t[t, :] = prev_tip_update_pos + delta_tip

        if isinstance(self.tip_angle_update_in_t, np.ndarray):  # if controlling also the tip angle
            if prev_tip_update_angle is None:
                prev_tip_update_angle = self.tip_angle_update_in_t[t-1]
            print('prev tip angle=', prev_tip_update_angle)
            print('torque on tip=', State.tip_torque)
            # delta_angle = self.alpha*(torque - current_tip_angle)*np.linalg.norm(self.loss) * (-0.5)  # old, not BEASTAL
            
            self.tip_angle_update_in_t[t] = prev_tip_update_angle + delta_angle
