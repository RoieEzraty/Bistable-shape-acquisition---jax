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
    def __init__(self, Strctr, T: int, desired_buckle_arr: jax.Array, sampling = 'Uniform') -> None:
        
        self.T = T  # total training set size (and algorithm time, not to confuse with time to reach equilibrium state)
        self.desired_buckle_arr = desired_buckle_arr
        self.desired_pos_in_t = np.zeros((Strctr.nodes, 2, T))
        self.desired_Fx_in_t = np.zeros(T)
        self.loss_in_t = np.zeros(T)

    def create_dataset(self, Strctr: "StructureClass", sampling: str) -> None:
        if sampling == 'Uniform':
            x_loc_in_t = np.random.uniform(0, Strctr.hinges*Strctr.L, size=self.T)
            y_loc_in_t = np.random.uniform(-Strctr.L, Strctr.L, size=self.T)
            self.tip_loc_in_t = np.stack(((x_loc_in_t), (y_loc_in_t.T)), axis=1)
        else:
            print('User specified incompatible sampling')

    def set_desired(self, pos_arr: jax.Array, Fy, t: int) -> None:
        self.desired_pos_in_t[:, :, t] = pos_arr
        self.desired_Fx_in_t[t] = Fy

    def calc_loss(self, Fx: float, t) -> None:
        self.loss = self.desired_Fx_in_t[t] - Fx
        self.loss_in_t[t] = self.loss
