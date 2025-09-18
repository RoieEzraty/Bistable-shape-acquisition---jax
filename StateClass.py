from __future__ import annotations

import time
import diffrax
import jax
import jax.numpy as jnp
import equinox as eqx
from jax import grad, jit, vmap
from jax.experimental.ode import odeint

from typing import Tuple, List
from numpy import array, zeros
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional

if TYPE_CHECKING:
    from StructureClass import StructureClass
    from VariablesClass import VariablesClass


# ===================================================
# Class - State Variables - node positions etc.
# ===================================================


class StateClass(eqx.Module):
    """
    Dynamic state of the chain (positions + hinge stiffness regime).
    """
    
    # ---- state / derived ----
    rest_lengths: jax.Array        # (H+1,) edge rest lengths (from initial pos)
    initial_hinge_angles: jax.Array  # (H,) hinge angles at rest (usually zeros)  
    pos_arr: jax.Array = eqx.field(init=False)   # (hinges+2, 2) integer coordinates
    buckle: jax.Array           # (H,) ∈ {+1,-1} per hinge/shim (direction of stiff side)
    
#     # calcaulted
#     k_rot_state: jax.Array  # (H,) effective hinge stiffnesses, soft or stiff, theta dependent
    
    def __init__(self, Strctr: "StructureClass", buckle: jax.Array):
        # default buckle: all +1
        if buckle is None:
            self.buckle = jnp.ones((Strctr.hinges, Strctr.shims), dtype=jnp.int32)
        else:
            self.buckle = buckle
            assert self.buckle.shape == (Strctr.hinges, Strctr.shims)
            
        self.pos_arr = self._initiate_pos(Strctr.hinges)  # (N=hinges+2, 2)
        # with a straight chain, each edge's rest length = init_spacing
        self.rest_lengths = jnp.full((Strctr.hinges + 1,), Strctr.L, dtype=jnp.float32)
        # straight chain -> 0 resting hinge angles
        self.initial_hinge_angles = jnp.zeros((Strctr.hinges,), dtype=jnp.float32)
            
    # --- build ---
    
    @staticmethod
    def _initiate_pos(hinges: int) -> jax.Array:
        """`(hinges+2, 2)` each pair is (xi, yi) of point i going like [[0, 0], [1, 0], [2, 0], etc]"""
        x = jnp.arange(hinges + 2, dtype=jnp.float32)
        return jnp.stack([x, jnp.zeros_like(x)], axis=1)
    
    @eqx.filter_jit
    def energy(self, Variabs: "VariablesClass", Strctr: "StructureClass", pos_arr: jnp.Array) -> jnp.Array[float]:
        """Compute the potential energy of the origami with the resting positions as reference"""

        thetas = vmap(lambda h: Strctr._get_theta(pos_arr, h))(jnp.arange(Strctr.hinges))
        # print(thetas)
        # jax.debug.print("thetas = {}", thetas)
        edges_length = vmap(lambda e: Strctr._get_edge_length(pos_arr, e))(jnp.arange(Strctr.edges))
        T = thetas[:, None]                 # (H,1)
        TH = Variabs.thetas_ss[:, None]      # (H,1)
        B = self.buckle                     # (H,S)

        # torques on hinges
        stiff_mask = ((B == 1) & (T > -TH)) | ((B == -1) & (T < TH))
        k_rot_state = jnp.where(stiff_mask, Variabs.k_stiff, Variabs.k_soft)  # (H,S)
        rotation_energy = 0.5 * jnp.sum(k_rot_state * (T - TH) ** 2)

        # stretch of material - should not stretch at all
        stretch_energy = 0.5 * jnp.sum(Variabs.k_stretch * (edges_length - Strctr.rest_lengths) ** 2)

        # total
        total_energy = rotation_energy + stretch_energy
        return jnp.array([total_energy, rotation_energy, stretch_energy])

    @eqx.filter_jit
    def total_potential_energy(self, variabs: "VariablesClass", strctr: "StructureClass",
                               pos_arr: jax.Array) -> jax.Array[jnp.float_]:
        return self.energy(variabs, strctr, pos_arr)[0]

    def reshape_pos_arr_2_state(pos_arr: jnp.Array[jnp.float_]) -> jnp.Array[jnp.float_]:
        first_half = pos_arr.flatten()
        second_half = jnp.zeros_like(first_half)
        return jnp.concatenate([first_half, second_half])

    def reshape_state_2_pos_arr(state: jnp.Array[jnp.float_], pos_arr) -> jnp.Array[jnp.float_]:
        return state.reshape(pos_arr.shape)
