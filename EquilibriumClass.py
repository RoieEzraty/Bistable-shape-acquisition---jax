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
    calc_through_energy: bool  # whether to calculate state through grad of energy or derictly w/forces

    # ---- state / derived ----
    rest_lengths: jax.Array                      # (H+1,) edge rest lengths (from initial pos)
    initial_hinge_angles: jax.Array              # (H,) hinge angles at rest (usually zeros)  
    init_pos: jax.Array = eqx.field(init=False)  # (hinges+2, 2) integer coordinates
    buckle_arr: jax.Array                        # (H,) ∈ {+1,-1} per hinge/shim (direction of stiff side)
    time_points: jax.Array                       # (T_eq, ) time steps for simulating equilibrium configuration
    
    def __init__(self, Strctr: "StructureClass", T: float, n_steps: int, damping_coeff: float, mass: float,
                 calc_through_energy: bool = True, buckle_arr: jax.Array = None, pos_arr: jax.Array = None):
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
            self.init_pos = helpers_builders._initiate_pos(Strctr.edges+1, Strctr.L)  # (N=hinges+2, 2)
        else:
            self.init_pos = jnp.asarray(pos_arr)
            
        # each edge's rest length is L, it's fixed and very stiff 
        self.rest_lengths = jnp.full((Strctr.hinges + 1,), Strctr.L, dtype=jnp.float32)
        # straight chain -> 0 resting hinge angles
        self.initial_hinge_angles = jnp.zeros((Strctr.hinges,), dtype=jnp.float32)

        # whether to calculate through grad of energy or through forces
        self.calc_through_energy = calc_through_energy

    def calculate_state(self, Variabs: "VariablesClass", Strctr: "StructureClass", control_first_edge: bool = True,
                        tip_pos: jax.Array = None, tip_angle: jax.float = None):
        
        n_coords = Strctr.n_coords                     # = 2 * (H+2)
        N = Strctr.hinges + 2                          # number of nodes
        last = N - 1

        # --- fixed and imposed DOFs initialize --- 
        fixed_DOFs = jnp.zeros((n_coords,), dtype=bool)
        imposed_DOFs = jnp.zeros((n_coords,), dtype=bool)

        # --- fixed: node 0, potentially also node 1 ---
        if control_first_edge:  
            nodes = [0, 1]  # two nodes are at 0
        else: 
            nodes = [0]  # first node at 0, 0
        for node in nodes:
            fixed_DOFs = fixed_DOFs.at[self.dof_idx(node, 0)].set(True)
            fixed_DOFs = fixed_DOFs.at[self.dof_idx(node, 1)].set(True)
        # fixed values (vector, not function)
        # fixed_vals = self.init_pos.reshape((-1,))  # (n_coords,)
        fixed_vals = jnp.zeros((n_coords,), dtype=float)
        fixed_vals = fixed_vals.at[fixed_DOFs].set(self.init_pos.reshape((-1))[fixed_DOFs])

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
                                                                 tip_angle=jnp.asarray(tip_angle,
                                                                                       dtype=self.init_pos.dtype),
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

    # --- energy ---
    @eqx.filter_jit
    def energy(self, Variabs: "VariablesClass", Strctr: "StructureClass", pos_arr: jnp.Array) -> jnp.Array[float]:
        """Compute the potential energy of the origami with the resting positions as reference"""

        thetas = vmap(lambda h: Strctr._get_theta(pos_arr, h))(jnp.arange(Strctr.hinges))
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
            theta_eff = B[:, None] * T              # (H,S)
            k_rot_state = Variabs.k(theta_eff)   # (H,S)
            k_rot_state = jax.lax.stop_gradient(Variabs.k(theta_eff))  # don't take derivative w.r.t k
            rotation_energy = 0.5 * jnp.sum(k_rot_state * (theta_eff - TH)**2)

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
    def total_potential_energy_free(self, Variabs: "VariablesClass", Strctr: "StructureClass", t: int,
                                    x_free: jax.Array, *, free_mask: jax.Array, fixed_mask: jax.Array,
                                    imposed_mask: jax.Array, fixed_vals: jax.Array, imposed_vals: jax.Array):
        fixed_vals_t = fixed_vals(t)
        imposed_vals_t = imposed_vals(t)
        x_full = helpers_builders._assemble_full(free_mask, fixed_mask, imposed_mask, x_free, fixed_vals_t,
                                                 imposed_vals_t)
        return self.total_potential_energy(Variabs, Strctr, x_full)

    def total_potential_force(self, Variabs: "VariablesClass", Strctr: "StructureClass", t: float,
                              x_free: jax.Array, *,
                              free_mask: jax.Array, fixed_mask: jax.Array, fixed_vals: jax.Array,
                              imposed_mask: jax.Array, imposed_vals: jax.Array):
        """
        Compute the **internal reaction force** on all DOFs (positions only) at time `t`
        using the **direct force model** (hinge torques from a spline + linear edge stretch),
        given only the FREE subset `x_free`. Fixed and imposed DOFs are reconstructed
        via the provided masks and callables.

        This path does **not** differentiate energy. It is implemented explicitly:
          - Hinge torque per hinge is accumulated over shims and mapped to DOFs via J^T.
          - Edge stretch force is k(ℓ−ℓ₀) along the unit edge direction.
          - Returns the full reaction force (size = n_coords), from which the caller
            typically takes `[free_mask]` for the ODE RHS.
        - The torque is mapped to DOFs with **+ J^T τ** (restoring).

        Parameters
        ----------
        Variabs : VariablesClass
            Holds mechanical parameters and the spline torque function `torque(theta_eff)`.
        Strctr : StructureClass
            Geometry/topology provider.
        t : float
            Current simulation time.
        x_free : jax.Array, shape: (sum(free_mask),)
        free_mask, fixed_mask, imposed_mask : jax.Array[bool], shape: (n_coords,)
        fixed_vals : Union[jax.Array, Callable[[float], jax.Array]]
            Either a constant vector of fixed positions (shape n_coords)
            or a callable returning that vector at time `t`.
        imposed_vals : Callable[[float], jax.Array]
            Callable returning the full vector of imposed positions at time `t`.

        Returns
        -------
        jax.Array, shape: (n_coords,)
            Internal reaction force on **all position DOFs** (no velocities), matching
            the sign used in the equations of motion:
                accel = (f_ext + f_internal - damping * xdot_free) / mass
        """
        # 1) Rebuild full x vector and reshape
        fixed_vals_t = fixed_vals(t) if callable(fixed_vals) else fixed_vals
        imposed_vals_t = imposed_vals(t)  # callable by construction above
        x_full = helpers_builders._assemble_full(free_mask, fixed_mask, imposed_mask, x_free, fixed_vals_t,
                                                 imposed_vals_t)
        pos_arr = helpers_builders._reshape_state_2_pos_arr(x_full, self.init_pos)

        # 2) Hinge torques: tau_hinges (H,)
        thetas = jax.vmap(lambda h: Strctr._get_theta(pos_arr, h))(jnp.arange(Strctr.hinges))  # (H,)
        B = self.buckle_arr  # (H,S)
        theta_eff = B * thetas[:, None]   # (H,S)
        # torque per shim
        tau_shims = - Variabs.torque(theta_eff)  # (H,S)
        # signed + summed per hinge
        tau_hinges = jnp.sum(B * tau_shims, axis=1)  # (H,)

        # 3) Jacobian of theta for each hinge: (H, n_coords)
        def theta_jac_of_h(h):
            def theta_of_x(x_flat):
                pa = x_flat.reshape(pos_arr.shape)
                return Strctr._get_theta(pa, h)
            return jax.jacrev(theta_of_x)(x_full[:Strctr.n_coords])  # (n_coords,)
        theta_jacs = jax.vmap(theta_jac_of_h)(jnp.arange(Strctr.hinges))  # (H, n_coords)

        # 4) Map torques to DOF forces
        F_theta_full = (theta_jacs.T @ tau_hinges).reshape(-1)            # (n_coords,)
        # 5) Edge stretch forces
        F_stretch_full = self.stretch_forces(Strctr, Variabs, pos_arr)         # (n_coords,)
        # Combine internal forces (reaction)
        F_internal_full = F_theta_full + F_stretch_full                   # (n_coords,)

        # jax.debug.print('F_theta_full {}', F_theta_full)
        # jax.debug.print('F_stretch_full {}', F_stretch_full)

        return F_internal_full

    @eqx.filter_jit
    def total_potential_force_from_full_x(self, Variabs: "VariablesClass", Strctr: "StructureClass",
                                          x_full: jax.Array):
        """
        Compute the **internal reaction force** on all DOFs (positions only) from a **full**
        position vector `x_full` using the **direct force model**, Does **not** differentiate energy.

        - Hinge torques: per-shim with buckle signs, summed per hinge, mapped via **+ J^T τ** (restoring).
        - Stretch: k(ℓ−ℓ₀) along unit edge; accumulate a += f, b -= f.
        - Useful for plotting and consistency checks against -∇U(x).

        Parameters
        ----------
        Variabs : VariablesClass
            Mechanical parameters and `torque(theta_eff)`.
        Strctr : StructureClass
            Geometry/topology provider.
        x_full : jax.Array, shape: (n_coords,)
            Full position vector (no velocities).

        Returns
        -------
        jax.Array, shape: (n_coords,)
            Internal reaction force on **all position DOFs**.
        """

        # x_full: (n_coords,) flattened positions ONLY (no velocities)
        pos_arr = helpers_builders._reshape_state_2_pos_arr(x_full, self.init_pos)

        # --- hinge torques per hinge (H,) ---
        thetas = jax.vmap(lambda h: Strctr._get_theta(pos_arr, h))(jnp.arange(Strctr.hinges))  # (H,)
        B = self.buckle_arr  # (H,S)
        theta_eff = B * thetas[:, None]  # (H,S)
        tau_shims = -Variabs.torque(theta_eff)  # (H,S)
        tau_hinges = jnp.sum(B * tau_shims, axis=1)  # (H,)

        # --- dense Jacobian (simple, OK for plotting; for scale use local-8DOF approach) ---
        def theta_jac_of_h(h):
            def theta_of_x(x_flat):
                pa = helpers_builders._reshape_state_2_pos_arr(x_flat, self.init_pos)
                return Strctr._get_theta(pa, h)
            return jax.jacrev(theta_of_x)(x_full)  # (n_coords,)
        theta_jacs = jax.vmap(theta_jac_of_h)(jnp.arange(Strctr.hinges))  # (H, n_coords)
        F_theta_full = (theta_jacs.T @ tau_hinges).reshape(-1)  # (n_coords,)

        # --- stretch forces ---
        F_stretch_full = self.stretch_forces(Strctr, Variabs, pos_arr)      # (n_coords,)

        # jax.debug.print('F_theta_full {}', F_theta_full)
        # jax.debug.print('F_stretch_full {}', F_stretch_full)

        # reaction/internal force on DOFs (same sign you use in rhs)
        return F_theta_full + F_stretch_full                                # (n_coords,)

    def potential_force_free(self, Variabs: "VariablesClass", Strctr: "StructureClass", t: float,
                             x_free: jax.Array, *,
                             free_mask: jax.Array, fixed_mask: jax.Array, fixed_vals: jax.Array,
                             imposed_mask: jax.Array, imposed_vals: jax.Array):
        """
        Internal reaction force on **FREE** DOFs only, for the ODE RHS.

        Mode:
        - If `self.calc_through_energy` is True: use `-∇U(x_full)` and slice by `free_mask`.
        - Else (direct-force): reconstruct `x_full` at time `t`, compute spline torque + stretch,
          then return `[free_mask]`.

        Parameters
        ----------
        Variabs : VariablesClass
        Strctr : StructureClass
        t : float
        x_free : jax.Array, shape: (sum(free_mask),)
        free_mask, fixed_mask, imposed_mask : jax.Array[bool], shape: (n_coords,)
        fixed_vals : Union[jax.Array, Callable[[float], jax.Array]]
        imposed_vals : Callable[[float], jax.Array]

        Returns
        -------
        jax.Array, shape: (sum(free_mask),)
            Internal reaction force on **free position DOFs** (restoring sign).
        """
        if self.calc_through_energy:
            return jax.grad(
                lambda xf: -self.total_potential_energy_free(
                    Variabs, Strctr, t, xf,
                    free_mask=free_mask, fixed_mask=fixed_mask, imposed_mask=imposed_mask, fixed_vals=fixed_vals,
                    imposed_vals=imposed_vals)
            )(x_free)
        else:  # no grad of energy, directly through forces instead
            return self.total_potential_force(Variabs, Strctr, t, x_free, free_mask=free_mask,
                                              fixed_mask=fixed_mask, imposed_mask=imposed_mask,
                                              fixed_vals=fixed_vals, imposed_vals=imposed_vals)[free_mask]
            
    def stretch_forces(self, Strctr: "StructureClass", Variabs: "VariablesClass",
                       pos_arr: NDArray[float]) -> jax.Array:
        """
        Linear **edge stretch** forces on all DOFs given node positions.

        For each edge e=(a,b):
          ℓ = ||p_b - p_a||, Δℓ = ℓ - ℓ₀, u = (p_b - p_a)/max(ℓ, eps)
          f = k_e * Δℓ * u
        Accumulate: a += f, b -= f (restoring).

        Parameters
        ----------
        Strctr : StructureClass
            Provides `edges_arr` (E,2) and `rest_lengths` (E,).
        Variabs : VariablesClass
            `k_stretch` scalar or (E,).
        pos_arr : jax.Array, shape: (N,2)

        Returns
        -------
        jax.Array, shape: (n_coords,)
            Flattened reaction force on all position DOFs.
        """
        # pos_arr: (N, 2)
        # Strctr.edge_list: (E, 2) with node indices (ia, ib)
        # Strctr.rest_lengths: (E,)
        # Variabs.k_stretch: scalar or (E,)
        edges = Strctr.edges_arr             # (E,2)
        pa = pos_arr[edges[:, 0], :]          # (E,2)
        pb = pos_arr[edges[:, 1], :]          # (E,2)
        d = pb - pa                          # (E,2)
        l = jnp.linalg.norm(d, axis=1)       # (E,)
        # Avoid divide-by-zero in early steps
        l_safe = jnp.where(l > 1e-12, l, 1e-12)
        dl = l - Strctr.rest_lengths          # (E,)

        k = Variabs.k_stretch
        if k.ndim == 0:
            k = jnp.full_like(dl, k)
        # force magnitude along edge direction
        fmag = k * dl / l_safe                 # (E,)
        fvec = (fmag[:, None]) * d             # (E,2)

        # accumulate to node forces
        F = jnp.zeros_like(pos_arr)           # (N,2)
        F = F.at[edges[:, 0], :].add(+fvec)
        F = F.at[edges[:, 1], :].add(-fvec)
        return F.reshape(-1)                  # (n_coords,)

    # -------- external forces (optional) ----------
    def force_function(self, t: float) -> jnp.ndarray:
        # No external forces; you can add tip forces here if needed
        return jnp.zeros((self.init_pos.size,), dtype=self.init_pos.dtype)

    def force_function_free(self,
                            t: float,
                            force_function: Callable[[float], jax.Array],
                            *,
                            free_mask: jax.Array) -> jax.Array:
        """External force restricted to free DOFs."""
        return force_function(t)[free_mask]


# # NOT IN USE

    # def calculate_energy_in_t(self, Variabs: "VariablesClass", Strctr: "StructureClass",
    #                           displacements: NDArray[np.float_]) -> jax.array:
    #     """
    #     Calculate energies pos-mortem
    #     """
    #     T = jnp.shape(displacements)[0]  # problem, displacements should be jax in that sense and not NDArray
    #     tot_energy_in_t = jnp.zeros(T)
    #     rot_energy_in_t = jnp.zeros(T)
    #     stretch_energy_in_t = jnp.zeros(T)
    #     jax.debug.print('T {}', T)
    #     for t in range(T):
    #         energs = self.energy(Variabs, Strctr, displacements[t])
    #         tot_energy_in_t[t], rot_energy_in_t[t], stretch_energy_in_t[t] = energs[0], energs[1], energs[2]
    #     return jnp.array([tot_energy_in_t, rot_energy_in_t, stretch_energy_in_t])

#     def check_force_sign(self, Variabs, Strctr, x_full, free_mask=None, eps=1e-9):
#         """
#         Compares F_direct (your torque+stretch implementation) to F_grad (restoring force from energy).
#         Returns useful scalar diagnostics. If free_mask is provided, the comparison is done on free DOFs only.
#         """
#         # Restoring force from energy: F_grad = -∇U
#         U = lambda x: self.total_potential_energy(Variabs, Strctr, x)
#         F_grad_full = jax.grad(lambda xf: -U(xf))(x_full)

#         # Your direct internal force (same sign convention used in rhs)
#         F_direct_full = self.total_potential_force_from_full_x(Variabs, Strctr, x_full)

#         if free_mask is not None:
#             Fg = F_grad_full[free_mask]
#             Fd = F_direct_full[free_mask]
#         else:
#             Fg = F_grad_full
#             Fd = F_direct_full

#         n_g = jnp.linalg.norm(Fg)
#         n_d = jnp.linalg.norm(Fd)
#         dot = jnp.dot(Fd, Fg)
#         cos = dot / (jnp.maximum(n_g * n_d, eps))
#         rel_l2_err = jnp.linalg.norm(Fd - Fg) / jnp.maximum(n_g, eps)
#         angle_deg = jnp.degrees(jnp.arccos(jnp.clip(cos, -1.0, 1.0)))
#         max_abs_diff = jnp.max(jnp.abs(Fd - Fg))

#         return {
#             "norm_F_grad": n_g,
#             "norm_F_direct": n_d,
#             "dot(F_direct, F_grad)": dot,
#             "cosine_similarity": cos,       # ~1.0 when aligned
#             "angle_deg": angle_deg,         # ~0° when aligned
#             "rel_L2_err_vs_grad": rel_l2_err,
#             "max_abs_diff": max_abs_diff,
#         }
# # 