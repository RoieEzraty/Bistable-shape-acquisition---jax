from __future__ import annotations
from IPython.display import HTML

import time
import copy
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

import dynamics, plot_funcs

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
    # pos_arr_in_t: jax.Array = eqx.field(default=None, init=False)   # (hinges+2, 2, t) integer coordinates
    # dynamics_last_step: jax.Array = eqx.field(default=None, init=False)
    # vel_last_step: jax.Array = eqx.field(default=None, init=False)
    # potential_force_last_step: jax.Array = eqx.field(default=None, init=False)
    buckle: jax.Array           # (H,) ∈ {+1,-1} per hinge/shim (direction of stiff side)
    
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

        # self.dynamics_last_step = jnp.zeros((self.pos_arr, t_for_dynamics), dtype=jnp.float32)
        # self.vel_last_step = jnp.full((Strctr.hinges + 1,), Strctr.L, dtype=jnp.float32)
        # self.potential_force_last_step = jnp.full((Strctr.hinges + 1,), Strctr.L, dtype=jnp.float32)

    def calculate_state(self, Variabs: "VariablesClass", Strctr: "StructureClass"):
        n_coords = self.pos_arr.size                # = 2 * (H+2)

        # -------- masks (boolean) ----------
        # We impose node 0 at (0,0). No other fixed DOFs.
        fixed_DOFs = jnp.zeros((n_coords,), dtype=bool)

        imposed_disp_DOFs = jnp.zeros((n_coords,), dtype=bool)
        imposed_disp_DOFs = imposed_disp_DOFs.at[self.dof_idx(0, 0)].set(True)   # x of node 0
        imposed_disp_DOFs = imposed_disp_DOFs.at[self.dof_idx(0, 1)].set(True)   # y of node 0
        imposed_disp_DOFs = imposed_disp_DOFs.at[self.dof_idx(1, 0)].set(True)   # x of node 0
        imposed_disp_DOFs = imposed_disp_DOFs.at[self.dof_idx(1, 1)].set(True)   # y of node 0

        # -------- initial state (positions & velocities) ----------
        x0 = self.pos_arr.flatten()                  # start from current geometry
        v0 = jnp.zeros_like(x0)                    # start at rest
        state_0 = jnp.concatenate([x0, v0], axis=0)

        # -------- time grid ----------
        dt = 1e-3
        t0, t1, n_steps = 0.0, 800.0, int(1/dt)
        time_points = jnp.linspace(t0, t1, n_steps)

        # -------- run dynamics ----------
        final_pos, pos_in_t, vel_in_t, potential_force_evolution = dynamics.solve_dynamics(
            time_points,
            state_0,
            Variabs,
            Strctr,
            self,
            force_function = None,
            fixed_DOFs = fixed_DOFs,
            imposed_disp_DOFs = imposed_disp_DOFs,
            imposed_disp_values = None
        )

        return final_pos, pos_in_t, vel_in_t, potential_force_evolution

        # self.pos_arr_in_t = final_pos
        # self.dynamics_last_step = pos_in_t
        # self.vel_last_step = vel_in_t
        # self.potential_force_last_step = potential_force_evolution
            
    # --- build ---
    @staticmethod
    def _initiate_pos(hinges: int) -> jax.Array:
        """`(hinges+2, 2)` each pair is (xi, yi) of point i going like [[0, 0], [1, 0], [2, 0], etc]"""
        x = jnp.arange(hinges + 2, dtype=jnp.float32)
        pos_arr = jnp.stack([x, jnp.zeros_like(x)], axis=1)
        # pos_arr_in_t = copy.copy(pos_arr)
        # return pos_arr, pos_arr_in_t
        return pos_arr

    # --- reshape ---
    @staticmethod
    def _assemble_full(x_free: jax.Array,
                       free_mask: jax.Array,       # bool (n_coords,)
                       fixed_mask: jax.Array,      # bool (n_coords,)
                       imposed_mask: jax.Array,    # bool (n_coords,)
                       imposed_vals_t: jax.Array,  # (n_coords,)
                       n_coords: int) -> jax.Array:
        """Build full flattened x from free DOFs + constraints at time t."""
        x_full = jnp.zeros((n_coords,), dtype=imposed_vals_t.dtype)
        x_full = x_full.at[free_mask].set(x_free)
        x_full = jnp.where(fixed_mask, 0.0, x_full)
        x_full = jnp.where(imposed_mask, imposed_vals_t, x_full)
        return x_full

    # Helper to map (node, component) -> flat DOF index
    # component: 0 = x, 1 = y
    @staticmethod
    def dof_idx(node: int, comp: int) -> int:
        return 2*node + comp

    # -------- imposed displacement function ----------
    # Must return a length-n_coords vector (flattened order) for any time t
    def imposed_disp_values(self, t: float) -> jnp.ndarray:
        vals = jnp.zeros((self.pos_arr.size,), dtype=self.pos_arr.dtype)
        # First node pinned at (0,0) → both DOFs are zero; nothing else imposed.
        # If you ever want a moving base, set these two entries to your function of t.
        return vals

    # -------- external forces (optional) ----------
    def force_function(self, t: float) -> jnp.ndarray:
        # No external forces; you can add tip forces here if needed
        return jnp.zeros((self.pos_arr.size,), dtype=self.pos_arr.dtype)

    @staticmethod
    def _reshape_pos_arr_2_state(pos_arr: jnp.Array[jnp.float_]) -> jnp.Array[jnp.float_]:
        first_half = pos_arr.flatten()
        second_half = jnp.zeros_like(first_half)
        return jnp.concatenate([first_half, second_half])

    @staticmethod
    def _reshape_state_2_pos_arr(state: jnp.Array[jnp.float_], pos_arr) -> jnp.Array[jnp.float_]:
        return state.reshape(pos_arr.shape)

    # --- energy ---
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
                               x_full: jax.Array) -> jax.Array[jnp.float_]:
        pos_arr = self._reshape_state_2_pos_arr(x_full, self.pos_arr)
        print('pos_arr in total_potential_energy ', pos_arr)
        return self.energy(variabs, strctr, pos_arr)[0]

    def total_potential_energy_free(self,
                                    Variabs: "VariablesClass",
                                    Strctr: "StructureClass",
                                    t: float,
                                    x_free: jax.Array,
                                    *,
                                    free_mask: jax.Array,
                                    fixed_mask: jax.Array,
                                    imposed_mask: jax.Array,
                                    imposed_disp_values: Callable[[float], jax.Array]
                                    ) -> jax.Array:
        """Total potential energy evaluated only on the free DOFs."""
        n_coords = self.pos_arr.size
        imp_vals_t = imposed_disp_values(t)                 # (n_coords,)
        x_full = self._assemble_full(x_free, free_mask, fixed_mask, imposed_mask,
                                     imp_vals_t, n_coords)
        return self.total_potential_energy(Variabs, Strctr, x_full)

    def potential_force_free(self,
                             Variabs: "VariablesClass",
                             Strctr: "StructureClass",
                             t: float,
                             x_free: jax.Array,
                             *,
                             free_mask: jax.Array,
                             fixed_mask: jax.Array,
                             imposed_mask: jax.Array,
                             imposed_disp_values: Callable[[float], jax.Array]
                             ) -> jax.Array:
        """-∂E/∂x on the free DOFs."""
        # jax.debug.print("imposed_disp_value inside potential_force_free = {}", imposed_disp_values)
        return jax.grad(
            lambda x: -self.total_potential_energy_free(
                Variabs, Strctr, t, x,
                free_mask=free_mask,
                fixed_mask=fixed_mask,
                imposed_mask=imposed_mask,
                imposed_disp_values=imposed_disp_values
            )
        )(x_free)

    def force_function_free(self,
                            t: float,
                            force_function: Callable[[float], jax.Array],
                            *,
                            free_mask: jax.Array) -> jax.Array:
        """External force restricted to free DOFs."""
        return force_function(t)[free_mask]
