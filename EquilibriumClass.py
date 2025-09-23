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
    Jax variables of the dynamic state of the chain (positions + hinge stiffness regime)

    Stores the reference geometry, rest lengths, and buckle states 
    of the system, and provides methods to compute equilibrium displacements, 
    energies, and forces given constraints and imposed displacements.
    The equilibrium configuration is such that the total energy is at a local minimum,
    reached by following a damped equation of motion for node displacements.
    Energy has hinge torque and edge stretch contributions, such that the stretch modulus is stiff,
    hence negligible strechability.

    Conventions
    -----------
    - Hinge angles are measured **counter-clockwise (CCW)**.
    - Buckle state:
        * `1`  → hinge buckled **downwards**
        * `-1` → hinge buckled **upwards**
    - The first two nodes (0 and 1) are fixed in space by default.

    Attributes
    ----------
    rest_lengths : jax.Array, shape (hinges+1,)
        Rest lengths of each edge, initialized from the straight configuration.
    initial_hinge_angles : jax.Array, shape (hinges,)
        Resting hinge angles (zero in straight configuration).
    init_pos : jax.Array, shape (nodes,2)
        Initial nodal positions of the chain (nodes = hinges+2).
    buckle_arr : jax.Array, shape (hinges ,shims)
        Buckle state for each hinge and shim (values in {+1, -1}).

    Methods
    -------
    calculate_state(Variabs, Strctr, tip_pos=None)
        Run a dynamic relaxation process to compute equilibrium shape
        under constraints and optional imposed tip displacement.
        Returns final positions, full trajectory, velocities, and potential forces.
    dof_idx(node, comp)
        Map a node index and component (0=x, 1=y) to a flat degree-of-freedom index.
    imposed_vals(t)
        Return imposed displacement vector at time `t` (default: none).
    force_function(t)
        Return external force vector at time `t` (default: zero).
    energy(Variabs, Strctr, pos_arr)
        Compute total, rotational, and stretching energies for a configuration.
    total_potential_energy(Variabs, Strctr, x_full)
        Compute scalar total potential energy from a flattened state vector.
    total_potential_energy_free(...)
        Same as above, but restricted to free DOFs given masks and imposed values.
    potential_force_free(...)
        Compute forces on free DOFs via gradient of total potential energy.
    force_function_free(t, force_function, free_mask)
        Restrict external forces to free DOFs only.
    """
    
    # ---- state / derived ----
    rest_lengths: jax.Array        # (H+1,) edge rest lengths (from initial pos)
    initial_hinge_angles: jax.Array  # (H,) hinge angles at rest (usually zeros)  
    init_pos: jax.Array = eqx.field(init=False)   # (hinges+2, 2) integer coordinates
    buckle_arr: jax.Array           # (H,) ∈ {+1,-1} per hinge/shim (direction of stiff side)
    
    def __init__(self, Strctr: "StructureClass", buckle_arr: jax.Array = None, pos_arr: jax.Array = None):
        # default buckle: all +1
        if buckle_arr is None:
            self.buckle_arr = jnp.ones((Strctr.hinges, Strctr.shims), dtype=jnp.int32)
        else:
            self.buckle_arr = buckle_arr
            assert self.buckle_arr.shape == (Strctr.hinges, Strctr.shims)
            
        if pos_arr is None:
            self.init_pos = helpers_builders._initiate_pos(Strctr.hinges)  # (N=hinges+2, 2)
        else:
            self.init_pos = pos_arr
        # with a straight chain, each edge's rest length = init_spacing
        self.rest_lengths = jnp.full((Strctr.hinges + 1,), Strctr.L, dtype=jnp.float32)
        # straight chain -> 0 resting hinge angles
        self.initial_hinge_angles = jnp.zeros((Strctr.hinges,), dtype=jnp.float32)

    def calculate_state(self, Variabs: "VariablesClass", Strctr: "StructureClass", tip_pos: jax.Array = None):
        
        n_coords = Strctr.n_coords                     # = 2 * (H+2)
        N = Strctr.hinges + 2                          # number of nodes
        last = N - 1

        # ---- fixed: nodes 0 and 1 ----
        fixed_DOFs = jnp.zeros((n_coords,), dtype=bool)
        for node in (0, 1):
            fixed_DOFs = fixed_DOFs.at[self.dof_idx(node, 0)].set(True)
            fixed_DOFs = fixed_DOFs.at[self.dof_idx(node, 1)].set(True)

        # constant vector (NOT a function)
        fixed_vals = self.init_pos.reshape((-1,))  # (n_coords,)

        imposed_DOFs = jnp.zeros((n_coords,), dtype=bool)

        # Build a callable (always), even if mask is all False.
        base_vec = fixed_vals                          # start from initial positions

        if tip_pos is None:
            imposed_vals = (lambda t, v=base_vec: v)
        else:
            tip_xy = jnp.asarray(tip_pos, dtype=self.init_pos.dtype).reshape((2,))
            idx_x = self.dof_idx(last, 0)
            idx_y = self.dof_idx(last, 1)
            imposed_DOFs = imposed_DOFs.at[idx_x].set(True).at[idx_y].set(True)
            imposed_arr = base_vec.at[idx_x].set(tip_xy[0]).at[idx_y].set(tip_xy[1])
            imposed_vals = (lambda t, v=imposed_arr: v)

        # -------- initial state (positions & velocities) ----------
        x0 = self.init_pos.flatten()                  # start from current geometry
        v0 = jnp.zeros_like(x0)                    # start at rest
        state_0 = jnp.concatenate([x0, v0], axis=0)

        # -------- time grid ----------
        dt = 1e-3
        t0, t1, n_steps = 0.0, 1600.0, int(1/dt)
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
            imposed_DOFs=imposed_DOFs,
            imposed_vals=imposed_vals,
        )

        return final_pos, pos_in_t, vel_in_t, potential_force_evolution

    # Helper to map (node, component) -> flat DOF index
    # component: 0 = x, 1 = y
    @staticmethod
    def dof_idx(node: int, comp: int) -> int:
        return 2*node + comp

    # -------- imposed displacement function ----------
    # Must return a length-n_coords vector (flattened order) for any time t
    def imposed_vals(self, t: float) -> jnp.ndarray:
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
        print('shape pos arr', jnp.shape(pos_arr))

        thetas = vmap(lambda h: Strctr._get_theta(pos_arr, h))(jnp.arange(Strctr.hinges))
        # print(thetas)
        # jax.debug.print("thetas = {}", thetas)
        edges_length = vmap(lambda e: Strctr._get_edge_length(pos_arr, e))(jnp.arange(Strctr.edges))
        T = thetas[:, None]                 # (H,1)
        TH = Variabs.thetas_ss[:, None]      # (H,1)
        B = self.buckle_arr                     # (H,S)
        
        # spring constant is position dependent
        if Variabs.k_type == 'Numerical':
            stiff_mask = ((B == 1) & (T < TH)) | ((B == -1) & (T > -TH))  # thetas are counter-clockwise    
            k_rot_state = jnp.where(stiff_mask, Variabs.k_stiff, Variabs.k_soft)  # (H,S)
            rotation_energy = 0.5 * jnp.sum(k_rot_state * (T - TH) ** 2)
        elif Variabs.k_type == 'Experimental':
            # k(theta) from the experimental curve; shape (H,)
            # k_theta = Variabs.k(thetas)                     # (H,)
            # k_rot_state = jnp.broadcast_to(k_theta, (Strctr.hinges, Strctr.shims))
            # per-shim effective angle: apply buckle sign to the angle, NOT to k
            theta_eff = B[:, None] * T              # (H,S)
            theta_ss_eff = B[:, None] * TH
            # theta_eff = T
            # theta_ss_eff = TH

            # evaluate experimental stiffness per shim
            k_rot_state = Variabs.k(theta_eff)   # (H,S)
            # jax.debug.print("k_rot_state shape = {}", k_rot_state.shape)
            # jax.debug.print("theta_ss_eff = {}", theta_ss_eff.shape)

            # rotation_energy = 0.5 * jnp.sum(k_rot_state * (T - theta_ss_eff) ** 2)

            # measure the error in the SAME signed frame
            delta = T - theta_ss_eff                # (H,S)

            rotation_energy = 0.5 * jnp.sum(k_rot_state * delta**2)

        # E_k = 1/2 * k * delta_theta ** 2
        # rotation_energy = 0.5 * jnp.sum(k_rot_state * (T - B * TH) ** 2)
        # rotation_energy = 0.5 * jnp.sum(k_rot_state * (T - TH) ** 2)

        # E_stretch = 1/2 * k_stretch * delta_l ** 2
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

    # EquilibriumClass.py
    def total_potential_energy_free(self, Variabs, Strctr, t, x_free, *,
                                    free_mask, fixed_mask, imposed_mask, fixed_vals,
                                    imposed_vals):
        fixed_vals_t = fixed_vals(t)
        imposed_vals_t = imposed_vals(t)
        x_full = helpers_builders._assemble_full(free_mask, fixed_mask, imposed_mask, x_free, fixed_vals_t, imposed_vals_t)
        return self.total_potential_energy(Variabs, Strctr, x_full)

    def potential_force_free(self, Variabs, Strctr, t, x_free, *,
                             free_mask, fixed_mask, fixed_vals, imposed_mask,
                             imposed_vals):
        return jax.grad(
            lambda xf: -self.total_potential_energy_free(
                Variabs, Strctr, t, xf,
                free_mask=free_mask, fixed_mask=fixed_mask, imposed_mask=imposed_mask, fixed_vals=fixed_vals,
                imposed_vals=imposed_vals)
        )(x_free)

    def force_function_free(self,
                            t: float,
                            force_function: Callable[[float], jax.Array],
                            *,
                            free_mask: jax.Array) -> jax.Array:
        """External force restricted to free DOFs."""
        return force_function(t)[free_mask]
