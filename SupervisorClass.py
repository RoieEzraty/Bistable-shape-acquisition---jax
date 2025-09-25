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
            x_pos_in_t = np.random.uniform((Strctr.hinges-1)*Strctr.L, Strctr.hinges*Strctr.L, size=self.T)
            y_pos_in_t = np.random.uniform(-Strctr.L/2, Strctr.L/2, size=self.T)
            self.tip_pos_in_t = np.stack(((x_pos_in_t), (y_pos_in_t.T)), axis=1)
            if isinstance(self.tip_angle_update_in_t, np.ndarray):  # if controlling also the tip angle
                self.tip_angle_in_t = np.random.uniform(-np.pi/4, np.pi/4, size=self.T)
        else:
            print('User specified incompatible sampling')

    def set_desired(self, pos_arr: jax.Array, Fx: float, Fy: float, t: int) -> None:
        self.desired_pos_in_t[:, :, t] = pos_arr
        self.desired_Fx_in_t[t] = Fx
        self.desired_Fy_in_t[t] = Fy

    def calc_loss(self, Fx: float, Fy: float, t) -> None:
        self.loss = np.array([self.desired_Fx_in_t[t] - Fx, self.desired_Fy_in_t[t] - Fy])
        self.loss_in_t[t, :] = self.loss

    def calc_update_tip(self, t: int, Fx: float, Fy: float, current_tip_pos: np.array, prev_tip_update_pos: np.array = None,
                        current_tip_angle: np.float = None, prev_tip_update_angle: np.float = None) -> None:
        if prev_tip_update_pos is None:
            prev_tip_update_pos = self.tip_pos_update_in_t[t-1, :]
        # delta_tip = self.alpha*(np.array([Fx, 0]) - prev_tip_update_pos)*(self.loss)
        delta_tip = self.alpha*(np.array([Fx, Fy]) - current_tip_pos)*(self.loss)*([1, 1])
        self.tip_pos_update_in_t[t, :] = prev_tip_update_pos + delta_tip

        if isinstance(self.tip_angle_update_in_t, np.ndarray):  # if controlling also the tip angle
            if prev_tip_update_angle is None:
                prev_tip_update_angle = self.tip_angle_update_in_t[t-1]
            torque = np.cos(current_tip_angle)*Fy-np.sin(current_tip_angle)*Fx
            print('prev tip angle=', prev_tip_update_angle)
            print('torque on tip=', torque)
            delta_angle = self.alpha*(torque - current_tip_angle)*np.linalg.norm(self.loss)
            self.tip_angle_update_in_t[t] = prev_tip_update_angle + delta_angle
