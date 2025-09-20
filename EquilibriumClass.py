from __future__ import annotations

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
    from VariablesClass import VariablesClass


# ===================================================
# Class - Calculate Equilibrium shape given forcing etc
# ===================================================


class EquilibriumClass(eqx.Module):
    """
    Dynamic state of the chain (positions + hinge stiffness regime).
    """
    
    # ---- state / derived ----
    rest_lengths: jax.Array        # (H+1,) edge rest lengths (from initial pos)
    initial_hinge_angles: jax.Array  # (H,) hinge angles at rest (usually zeros)  
    init_pos: jax.Array = eqx.field(init=False)   # (hinges+2, 2) integer coordinates
    buckle: jax.Array           # (H,) ∈ {+1,-1} per hinge/shim (direction of stiff side)
    # pos_arr_in_t: jax.Array = eqx.field(default=None, init=False)   # (hinges+2, 2, t) integer coordinates
    # dynamics_last_step: jax.Array = eqx.field(default=None, init=False)
    # vel_last_step: jax.Array = eqx.field(default=None, init=False)
    # potential_force_last_step: jax.Array = eqx.field(default=None, init=False)
    
    def __init__(self, Strctr: "StructureClass", buckle: jax.Array = None, pos_arr: jax.Array = None):
        # default buckle: all +1
        if buckle is None:
            self.buckle = jnp.ones((Strctr.hinges, Strctr.shims), dtype=jnp.int32)
        else:
            self.buckle = buckle
            assert self.buckle.shape == (Strctr.hinges, Strctr.shims)
            
        if pos_arr is None:
            self.init_pos = helpers_builders._initiate_pos(Strctr.hinges)  # (N=hinges+2, 2)
        else:
            self.init_pos = pos_arr
        # with a straight chain, each edge's rest length = init_spacing
        self.rest_lengths = jnp.full((Strctr.hinges + 1,), Strctr.L, dtype=jnp.float32)
        # straight chain -> 0 resting hinge angles
        self.initial_hinge_angles = jnp.zeros((Strctr.hinges,), dtype=jnp.float32)

    def calculate_state(self, Variabs: "VariablesClass", Strctr: "StructureClass", tip_loc: jax.Array = None):
        
        n_coords = Strctr.n_coords                     # = 2 * (H+2)
        N = Strctr.hinges + 2                          # number of nodes
        last = N - 1

        # ---------- masks ----------
        fixed_DOFs = jnp.zeros((n_coords,), dtype=bool)
        # fix node 0 (x,y)
        fixed_DOFs = fixed_DOFs.at[self.dof_idx(0, 0)].set(True)
        fixed_DOFs = fixed_DOFs.at[self.dof_idx(0, 1)].set(True)
        # fix node 1 (x,y)
        fixed_DOFs = fixed_DOFs.at[self.dof_idx(1, 0)].set(True)
        fixed_DOFs = fixed_DOFs.at[self.dof_idx(1, 1)].set(True)

        # fixed values = initial positions (flattened)
        fixed_vals = self.init_pos.flatten() 

        imposed_disp_DOFs = jnp.zeros((n_coords,), dtype=bool)

        # Build a callable (always), even if mask is all False.
        base_vec = fixed_vals                          # start from initial positions

        if tip_loc is None:
            imposed_disp_DOFs = jnp.zeros((n_coords,), bool)
            base_vec = fixed_vals
            imposed_disp_values = (lambda t, v=base_vec: v)
        else:
            tip_xy = jnp.asarray(tip_loc, dtype=self.init_pos.dtype).reshape((2,))
            idx_x = self.dof_idx(last, 0)
            idx_y = self.dof_idx(last, 1)
            imposed_disp_DOFs = (jnp.zeros((n_coords,), bool)
                                 .at[idx_x].set(True).at[idx_y].set(True))
            targ_vec = base_vec.at[idx_x].set(tip_xy[0]).at[idx_y].set(tip_xy[1])
            imposed_disp_values = (lambda t, v=targ_vec: v)

        # -------- initial state (positions & velocities) ----------
        x0 = self.init_pos.flatten()                  # start from current geometry
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
            force_function=None,
            fixed_DOFs=fixed_DOFs,
            fixed_vals=fixed_vals,
            imposed_disp_DOFs=imposed_disp_DOFs,
            imposed_disp_values=imposed_disp_values,
        )

        return final_pos, pos_in_t, vel_in_t, potential_force_evolution

        # self.init_pos_in_t = final_pos
        # self.dynamics_last_step = pos_in_t
        # self.vel_last_step = vel_in_t
        # self.potential_force_last_step = potential_force_evolution

    # Helper to map (node, component) -> flat DOF index
    # component: 0 = x, 1 = y
    @staticmethod
    def dof_idx(node: int, comp: int) -> int:
        return 2*node + comp

    # -------- imposed displacement function ----------
    # Must return a length-n_coords vector (flattened order) for any time t
    def imposed_disp_values(self, t: float) -> jnp.ndarray:
        vals = jnp.zeros((self.init_pos.size,), dtype=self.init_pos.dtype)
        # First node pinned at (0,0) → both DOFs are zero; nothing else imposed.
        # If you ever want a moving base, set these two entries to your function of t.
        return vals

    # -------- external forces (optional) ----------
    def force_function(self, t: float) -> jnp.ndarray:
        # No external forces; you can add tip forces here if needed
        return jnp.zeros((self.init_pos.size,), dtype=self.init_pos.dtype)

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
        pos_arr = helpers_builders._reshape_state_2_pos_arr(x_full, self.init_pos)
        return self.energy(variabs, strctr, pos_arr)[0]

    # def total_potential_energy_free(self,
    #                                 Variabs: "VariablesClass",
    #                                 Strctr: "StructureClass",
    #                                 t: float,
    #                                 x_free: jax.Array,
    #                                 *,
    #                                 free_mask: jax.Array,
    #                                 fixed_mask: jax.Array,
    #                                 imposed_mask: jax.Array,
    #                                 imposed_disp_values: Callable[[float], jax.Array]
    #                                 ) -> jax.Array:
    #     """Total potential energy evaluated only on the free DOFs."""
    #     # n_coords = self.init_pos.size
    #     imp_vals_t = imposed_disp_values(t)                 # (n_coords,)
    #     x_full = helpers_builders._assemble_full(x_free, free_mask, fixed_mask, imposed_mask, imp_vals_t, Strctr.n_coords)
    #     return self.total_potential_energy(Variabs, Strctr, x_full)

    # def potential_force_free(self,
    #                          Variabs: "VariablesClass",
    #                          Strctr: "StructureClass",
    #                          t: float,
    #                          x_free: jax.Array,
    #                          *,
    #                          free_mask: jax.Array,
    #                          fixed_mask: jax.Array,
    #                          imposed_mask: jax.Array,
    #                          imposed_disp_values: Callable[[float], jax.Array]) -> jax.Array:
    #     """-∂E/∂x on the free DOFs."""
    #     # jax.debug.print("imposed_disp_value inside potential_force_free = {}", imposed_disp_values)
    #     return jax.grad(lambda x: -self.total_potential_energy_free(Variabs, 
    #                                                                 Strctr, t, x,
    #                                                                 free_mask=free_mask,
    #                                                                 fixed_mask=fixed_mask,
    #                                                                 imposed_mask=imposed_mask,
    #                                                                 imposed_disp_values=imposed_disp_values))(x_free)

    # EquilibriumClass.py
    def total_potential_energy_free(self, Variabs, Strctr, t, x_free, *,
                                    free_mask, fixed_mask, fixed_vals, imposed_mask,
                                    imposed_disp_values):
        imp_vals_t = imposed_disp_values(t)                          # (nb,)
        x_full = helpers_builders._assemble_full(x_free, free_mask, fixed_mask, fixed_vals, imposed_mask, imp_vals_t)
        return self.total_potential_energy(Variabs, Strctr, x_full)

    def potential_force_free(self, Variabs, Strctr, t, x_free, *,
                             free_mask, fixed_mask, fixed_vals, imposed_mask,
                             imposed_disp_values):
        return jax.grad(
            lambda xf: -self.total_potential_energy_free(
                Variabs, Strctr, t, xf,
                free_mask=free_mask, fixed_mask=fixed_mask,
                fixed_vals=fixed_vals, imposed_mask=imposed_mask,
                imposed_disp_values=imposed_disp_values)
        )(x_free)

    def force_function_free(self,
                            t: float,
                            force_function: Callable[[float], jax.Array],
                            *,
                            free_mask: jax.Array) -> jax.Array:
        """External force restricted to free DOFs."""
        return force_function(t)[free_mask]
