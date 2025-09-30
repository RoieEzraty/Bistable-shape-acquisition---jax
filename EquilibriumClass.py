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
    T: float              
        End time for simulation of equilibrium state
    n_steps
        Number of steps per simulation of equilibrium state
    damping_coeff: float             
        Coefficient for right hand side of eqn of motion
    mass: float                      
        Newtonian mass for right hand side of eqn of motion

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
    
    # --- User input ---
    damping_coeff: float  # damping coefficient for right hand side of eqn of motion
    mass: float           # Newtonian mass for right hand side of eqn of motion

    # ---- state / derived ----
    rest_lengths: jax.Array                      # (H+1,) edge rest lengths (from initial pos)
    initial_hinge_angles: jax.Array              # (H,) hinge angles at rest (usually zeros)  
    init_pos: jax.Array = eqx.field(init=False)  # (hinges+2, 2) integer coordinates
    buckle_arr: jax.Array                        # (H,) ∈ {+1,-1} per hinge/shim (direction of stiff side)
    time_points: jax.Array                       # (T_eq, ) time steps for simulating equilibrium configuration
    
    def __init__(self, Strctr: "StructureClass", T: float, n_steps: int, damping_coeff: float, mass: float,
                 buckle_arr: jax.Array = None, pos_arr: jax.Array = None):
        self.damping_coeff = damping_coeff
        self.mass = mass
        self.time_points = jnp.linspace(0, T, n_steps)

        # default buckle: all +1
        if buckle_arr is None:
            self.buckle_arr = jnp.ones((Strctr.hinges, Strctr.shims), dtype=jnp.int32)
        else:
            self.buckle_arr = buckle_arr
            assert self.buckle_arr.shape == (Strctr.hinges, Strctr.shims)
            
        if pos_arr is None:
            self.init_pos = helpers_builders._initiate_pos(Strctr.hinges)  # (N=hinges+2, 2)
        else:
            self.init_pos = jnp.asarray(pos_arr)
            
        # each edge's rest length is L, it's fixed and very stiff 
        self.rest_lengths = jnp.full((Strctr.hinges + 1,), Strctr.L, dtype=jnp.float32)
        # straight chain -> 0 resting hinge angles
        self.initial_hinge_angles = jnp.zeros((Strctr.hinges,), dtype=jnp.float32)

    def calculate_state(self, Variabs: "VariablesClass", Strctr: "StructureClass", tip_pos: jax.Array = None,
                        tip_angle: jax.float = None):
        
        n_coords = Strctr.n_coords                     # = 2 * (H+2)
        N = Strctr.hinges + 2                          # number of nodes
        last = N - 1

        # --- fixed and imposed DOFs initialize --- 
        fixed_DOFs = jnp.zeros((n_coords,), dtype=bool)
        imposed_DOFs = jnp.zeros((n_coords,), dtype=bool)

        # --- fixed: nodes 0 and 1 ---
        for node in (0, 1):
            fixed_DOFs = fixed_DOFs.at[self.dof_idx(node, 0)].set(True)
            fixed_DOFs = fixed_DOFs.at[self.dof_idx(node, 1)].set(True)
        # fixed values (vector, not function)
        fixed_vals = self.init_pos.reshape((-1,))  # (n_coords,)

        # Build a callable (always), even if mask is all False.
        base_vec = fixed_vals  # start from initial positions
        imposed_arr = base_vec

        # imposed tip values
        if tip_pos is None:
            imposed_vals = (lambda t, v=base_vec: v)
        else:
            # set tip indices as true 
            idx_x = self.dof_idx(last, 0)
            idx_y = self.dof_idx(last, 1)
            imposed_DOFs = imposed_DOFs.at[idx_x].set(True).at[idx_y].set(True)
            # set tip values for imposed_vals
            tip_xy = jnp.asarray(tip_pos, dtype=self.init_pos.dtype).reshape((2,))
            imposed_arr = base_vec.at[idx_x].set(tip_xy[0]).at[idx_y].set(tip_xy[1])
            # imposed_vals = (lambda t, v=imposed_arr: v)

        # imposed tip angle amounts to fixing one node before last
        if tip_angle is None:
            pass
        else:
            if tip_pos is None:
                print('no tip angle could be imposed without tip loc, skipping tip angle')
            else:
                # set before tip indices as true
                idx_x = self.dof_idx(last-1, 0)
                idx_y = self.dof_idx(last-1, 1)
                imposed_DOFs = imposed_DOFs.at[idx_x].set(True).at[idx_y].set(True)
                # set before tip values for imposed vals
                before_tip_xy = helpers_builders._get_before_tip(tip_pos=tip_xy,        # from above
                                                                 tip_angle=jnp.asarray(tip_angle, dtype=self.init_pos.dtype),
                                                                 L=Strctr.L,
                                                                 dtype=self.init_pos.dtype)
                imposed_arr = imposed_arr.at[idx_x].set(before_tip_xy[0]).at[idx_y].set(before_tip_xy[1])
        imposed_vals = (lambda t, v=imposed_arr: v)

        # -------- initial state (positions & velocities) ----------
        x0 = self.init_pos.flatten()                  # start from current geometry
        v0 = jnp.zeros_like(x0)                    # start at rest
        state_0 = jnp.concatenate([x0, v0], axis=0)

        # -------- run dynamics ----------
        final_pos, pos_in_t, vel_in_t, potential_force_evolution = dynamics.solve_dynamics(
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

    def calculate_energy_in_t(self, Variabs, Strctr, displacements):
        T = jnp.shape(displacements)[0]
        tot_energy_in_t = jnp.zeros(T)
        rot_energy_in_t = jnp.zeros(T)
        stretch_energy_in_t = jnp.zeros(T)
        jax.debug.print('T {}', T)
        for t in range(T):
            energs = self.energy(Variabs, Strctr, displacements[t])
            tot_energy_in_t[t], rot_energy_in_t[t], stretch_energy_in_t[t] = energs[0], energs[1], energs[2]
        return jnp.array([tot_energy_in_t, rot_energy_in_t, stretch_energy_in_t])

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

        thetas = vmap(lambda h: Strctr._get_theta(pos_arr, h))(jnp.arange(Strctr.hinges))
        # print(thetas)
        # jax.debug.print("thetas = {}", thetas)
        edges_length = vmap(lambda e: Strctr._get_edge_length(pos_arr, e))(jnp.arange(Strctr.edges))
        T = thetas[:, None]                 # (H,1)
        TH = Variabs.thetas_ss[:, None]      # (H,1)
        B = self.buckle_arr                     # (H,S)
        TH_eff = B[:, None] * TH
        
        # spring constant is position dependent
        if Variabs.k_type == 'Numerical':
            stiff_mask = ((B == 1) & (T < TH)) | ((B == -1) & (T > -TH))  # thetas are counter-clockwise    
            k_rot_state = jnp.where(stiff_mask, Variabs.k_stiff, Variabs.k_soft)  # (H,S)
            rotation_energy = 0.5 * jnp.sum(k_rot_state * (T - TH_eff) ** 2)
        elif Variabs.k_type == 'Experimental':
            # k(theta) from the experimental curve; shape (H,)
            # k_theta = Variabs.k(thetas)                     # (H,)
            # k_rot_state = jnp.broadcast_to(k_theta, (Strctr.hinges, Strctr.shims))
            # per-shim effective angle: apply buckle sign to the angle, NOT to k
            theta_eff = B[:, None] * T              # (H,S)
            
            # theta_eff = T
            # theta_ss_eff = TH

            # evaluate experimental stiffness per shim
            k_rot_state = Variabs.k(theta_eff)   # (H,S)
            # jax.debug.print("k_rot_state shape = {}", k_rot_state.shape)
            # jax.debug.print("theta_ss_eff = {}", theta_ss_eff.shape)

            # rotation_energy = 0.5 * jnp.sum(k_rot_state * (T - theta_ss_eff) ** 2)
            rotation_energy = 0.5 * jnp.sum(k_rot_state * (T - TH_eff)**2)

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
    def total_potential_energy(self, Variabs: "VariablesClass", Strctr: "StructureClass",
                               x_full: jax.Array) -> jax.Array[jnp.float_]:
        pos_arr = helpers_builders._reshape_state_2_pos_arr(x_full, self.init_pos)
        return self.energy(Variabs, Strctr, pos_arr)[0]

    # EquilibriumClass.py
    def total_potential_energy_free(self, Variabs: "VariablesClass", Strctr: "StructureClass", t: int, x_free: jax.Array, *,
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
