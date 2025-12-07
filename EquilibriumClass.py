from __future__ import annotations

import copy
import diffrax
import time
import jax
import jax.numpy as jnp
import equinox as eqx
from jax import grad, jit, vmap
from jax.experimental.ode import odeint

from typing import Tuple, List
from numpy import array, zeros
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional

import dynamics, helpers_builders

if TYPE_CHECKING:
    from StructureClass import StructureClass
    from VariablesClass import VariablesClass
    from SupervisorClass import SupervisorClass
    from config import ExperimentConfig


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
    damping_coeff: float             
        Coefficient for right hand side of eqn of motion
    mass: float                      
        Newtonian mass for right hand side of eqn of motion

    rest_lengths : jax.Array, shape (hinges+1,)
        Rest lengths of each edge, initialized from the straight configuration.
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
    damping_coeff: float       # damping coefficient for right hand side of eqn of motion
    mass: float                # Newtonian mass for right hand side of eqn of motion
    tolerance: float           # tolerance for dynamics simulation step size
    calc_through_energy: bool  # whether to calculate state through grad of energy or derictly w/forces
    ramp_pos: bool             # if True, ramp imposed vals during equilibrium simulation from initials vals to final imposed
    rand_key: int              # random key for noise on DOFs during equilibrium calculation
    pos_noise: float           # noise amplitude on initial positions
    vel_noise: float           # noise amplitude on initial velocities

    # ---- state / derived ----
    rest_lengths: jax.Array                      # (H+1,) edge rest lengths (from initial pos)
    init_pos: jax.Array = eqx.field(init=False)  # (hinges+2, 2) integer coordinates
    buckle_arr: jax.Array                        # (H,) ∈ {+1,-1} per hinge/shim (direction of stiff side)
    time_points: jax.Array                       # (T_eq, ) time steps for simulating equilibrium configuration
    
    def __init__(self, Strctr: "StructureClass", CFG: ExperimentConfig, ramp_pos: bool = True, buckle_arr: jax.Array = None,
                 pos_arr: jax.Array = None):
        self.damping_coeff = CFG.Eq.damping
        self.mass = CFG.Eq.mass        
        self.time_points = jnp.linspace(0, CFG.Eq.T_eq, int(1e3))
        self.tolerance = CFG.Eq.tolerance

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
        self.calc_through_energy = CFG.Eq.calc_through_energy
        self.ramp_pos = CFG.Eq.ramp_pos
        self.rand_key = CFG.Eq.rand_key_Eq
        self.pos_noise = CFG.Eq.pos_noise
        self.vel_noise = CFG.Eq.vel_noise

    def calculate_state(self, Variabs: "VariablesClass", Strctr: "StructureClass", Sprvsr: "SupervisorClass",
                        init_pos: NDArray[float], control_first_edge: bool = True,  tip_pos: jax.Array | None = None,
                        tip_angle: float | jax.Array | None = None, pos_noise: float | jax.Array | None = None,
                        vel_noise: float | jax.Array | None = None) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Compute the equilibrium state of the chain given boundary conditions and optional noise.

        This function sets fixed and imposed DOFs (degrees of freedom) for the chain,
        builds the initial state (positions + velocities), and integrates the damped
        equations of motion until an equilibrium configuration is reached.

        Parameters
        ----------
        Variabs : VariablesClass
        Strctr : StructureClass
        control_first_edge : bool, optional
            If True (default), both node 0 and node 1 are fixed in space (first edge clamped).
            If False, only node 0 is fixed.
        tip_pos : jax.Array, optional
            Prescribed tip position as a 1D array of shape (2,) = (x_tip, y_tip).
            If None, the tip is free in position (unless constrained elsewhere).
        tip_angle : float or jax.Array, optional
            Prescribed tip angle (radians, CCW from +x). If provided together with
            `tip_pos`, this is enforced by fixing the node before the tip such that
            its position corresponds to a segment of length `Strctr.L` at angle `tip_angle`.
            If None, no tip-angle constraint is imposed.
        pos_noise : float or jax.Array, optional
            Amplitude (or full array) of **position noise** added to the initial positions.
            A uniform random noise ∈ U[-1, 1] is sampled per DOF and multiplied by
            `pos_noise`. Noise is applied only to **interior nodes**, i.e. all nodes
            except the first two (0, 1) and the last two (N-2, N-1). Boundary nodes
            remain exactly at their initial positions.
        vel_noise : float or jax.Array, optional
            Additive noise for the initial velocities. If provided, it is added to
            the zero-velocity initial condition (can be a scalar or an array
            broadcastable to the velocity vector).

        Returns
        -------
        final_pos : jax.Array, shape (N, 2)
            Final nodal positions at the end of the dynamic relaxation.
        pos_in_t : jax.Array, shape (T_eq_samples, N, 2)
            Time history of nodal positions over the integration time grid.
        vel_in_t : jax.Array, shape (T_eq_samples, N, 2)
            Time history of nodal velocities over the integration time grid.
        potential_force_in_t : jax.Array, shape (T_eq_samples, n_coords)
            Time history of the internal reaction forces (stretch + bending)
            on each positional DOF, evaluated along the trajectory.

        Notes
        -----
        - The fixed and imposed DOFs are enforced directly inside `dynamics.solve_dynamics`.
        - When `tip_angle` is imposed, the node before the tip is constrained so that
          the last segment has length `Strctr.L` and orientation `tip_angle`.
        - Positional noise is applied **after** flattening `init_pos` and **before**
          concatenating with the velocity vector.
        """
        init_pos = helpers_builders.numpy2jax(init_pos)

        # ------ fixed values (vector, not function) ------
        fixed_vals = self._set_fixed_vals(Strctr.fixed_mask)

        # ------ imposed tip position and possibly angle ------
        # Build a callable (always), even if mask is all False.
        imposed_vals = self._set_imposed_vals(Strctr, Sprvsr, Sprvsr.imposed_mask, tip_pos, tip_angle, fixed_vals, init_pos)

        # -------- initial state (positions & velocities) ----------
        pos_noise = Strctr.L * self.pos_noise  # scale relative to length
        vel_noise = 1.5 * self.vel_noise  # from equating 1/2mv^2 = mean(Torque)*L
        state_0 = helpers_builders._extend_pos_to_x0_v0(init_pos, pos_noise, vel_noise, self.rand_key)

        # -------- run dynamics ----------
        final_pos, pos_in_t, vel_in_t, potential_F_in_t = self.solve_dynamics(state_0, Variabs, Strctr,
                                                                              fixed_mask=Strctr.fixed_mask, 
                                                                              fixed_vals=fixed_vals,
                                                                              imposed_mask=Sprvsr.imposed_mask,
                                                                              imposed_vals=imposed_vals)

        # self.init_pos = pos_in_t[-1]

        # # split to components if you want:
        # F_stretch = self.stretch_forces(Strctr, Variabs, final_pos)     # (n_coords,)
        # F_theta = potential_F_in_t[-1] - F_stretch           # (n_coords,)

        # # reshape for per-node view
        # F_stretch_2d = F_stretch.reshape(-1, 2)
        # F_theta_2d = F_theta.reshape(-1, 2)
        # F_compare = jnp.hstack([F_stretch_2d, F_theta_2d])

        # print("\n=== Final-step per-node forces comparison ===")
        # print("(Fx_stretch, Fy_stretch,  Fx_theta, Fy_theta)")
        # print(F_compare)
        # print("\n=== total forces")
        # print(jnp.sum(F_compare, axis=1))

        return final_pos, pos_in_t, vel_in_t, potential_F_in_t[-1]

    def _set_fixed_vals(self, fixed_mask):
        # USED
        fixed_vals = jnp.zeros((len(fixed_mask),), dtype=float)
        return fixed_vals.at[fixed_mask].set(self.init_pos.reshape((-1,))[fixed_mask])

    # def _set_imposed_vals(self, Strctr: "StructureClass", Sprvsr: "SupervisorClass", imposed_mask, tip_pos, tip_angle,
    #                       fixed_vals):
    #     # USED
    #     # # Must return a length-n_coords vector (flattened order) for any time t
    #     base_vec = fixed_vals  # start from initial positions
    #     imposed_arr = base_vec

    #     # fill in using DOFs and vals
    #     if tip_pos is None:
    #         return (lambda t, v=base_vec: v)
    #     else:
    #         tip_xy = jnp.asarray(tip_pos, dtype=self.init_pos.dtype).reshape((2,))
    #         if tip_angle is None:
    #             imposed_arr = imposed_arr.at[imposed_mask].set(tip_xy)
    #         else:
    #             before_tip_xy = helpers_builders._get_before_tip(tip_pos=tip_xy,
    #                                                              tip_angle=jnp.asarray(tip_angle, dtype=self.init_pos.dtype),
    #                                                              L=Strctr.L, dtype=self.init_pos.dtype)
    #             imposed_arr = imposed_arr.at[Sprvsr.imposed_mask].set(jnp.concatenate([before_tip_xy, tip_xy]))
    #     #     vals = jnp.zeros((self.init_pos.size,), dtype=self.init_pos.dtype)
    #     #     # First node pinned at (0,0) → both DOFs are zero; nothing else imposed.
    #     #     # If you ever want a moving base, set these two entries to your function of t.
    #     #     return vals
    #         return (lambda t, v=imposed_arr: v)

    def _set_imposed_vals(self, Strctr: "StructureClass", Sprvsr: "SupervisorClass", imposed_mask, tip_pos, tip_angle, 
                          fixed_vals, init_pos):
        """
        Build a time-dependent imposed displacement function.

        - At t = 0: imposed DOFs equal the previous equilibrium positions (self.init_pos).
        - For 0 < t < T_ramp: linearly ramp to the new tip pose.
        - For t >= T_ramp: imposed DOFs stay at the new pose.
        """
        # ------ instantiate sizes ------
        # Full starting geometry (previous equilibrium), flattened
        start_vec = init_pos.reshape(-1)  # (n_coords,)

        # If there's nothing to impose, keep the previous equilibrium for all t
        if tip_pos is None:
            return (lambda t, v=start_vec: v)

        # Build target vector: same as start_vec, but with tip / before-tip overwritten
        target_vec = start_vec

        # jaxify tip coordiantes 
        tip_xy = jnp.asarray(tip_pos, dtype=init_pos.dtype).reshape((2,))

        # ------ final position as vector ------
        if tip_angle is None:  # Only tip position is imposed:
            target_vec = target_vec.at[imposed_mask].set(tip_xy)
        else:  # Tip position + tip angle: impose node before tip as well
            before_tip_xy = helpers_builders._get_before_tip(
                tip_pos=tip_xy,
                tip_angle=jnp.asarray(tip_angle, dtype=init_pos.dtype),
                L=Strctr.L,
                dtype=init_pos.dtype,
            )
            tip_vals = jnp.concatenate([before_tip_xy, tip_xy])  # (4,)
            # Sprvsr.imposed_mask should correspond to [before_tip_xy, tip_xy]
            target_vec = target_vec.at[Sprvsr.imposed_mask].set(tip_vals)

        # ------- position as temporal function ------
        if self.ramp_pos:  # linear ramp in time 
            T_total = self.time_points[-1]
            ramp_fraction = 0.25  # use half of the simulation time for the ramp
            T_ramp = ramp_fraction * T_total

            def imposed_vals(t, start=start_vec, target=target_vec, T_r=T_ramp):
                s = jnp.clip(t / T_r, 0.0, 1.0)  # Linear ramp parameter s(t) in [0, 1]
                return (1.0 - s) * start + s * target
        else:  # constant
            imposed_vals = lambda t, v=target_vec: v

        return imposed_vals

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
        x_full = helpers_builders._assemble_full_from_free(free_mask, fixed_mask, imposed_mask, x_free, fixed_vals_t,
                                                           imposed_vals_t)
        return self.total_potential_energy(Variabs, Strctr, x_full)

    def total_potential_force(self, Variabs: "VariablesClass", Strctr: "StructureClass", t: float,
                              x_free: jax.Array, *,
                              free_mask: jax.Array, fixed_mask: jax.Array, fixed_vals: jax.Array,
                              imposed_mask: jax.Array, imposed_vals: jax.Array):
        """
        USED from potential force free
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
        x_free : jax.Array of free node positions, x and y, shape: (2*nodes - 2*n_fixed - 2*n_imposed,)
        free_mask, fixed_mask, imposed_mask : jax.Array[bool], shape: (2*nodes,)
        fixed_vals : Union[jax.Array, Callable[[float], jax.Array]]
            Either a constant vector of fixed positions (shape 2*nodes)
            or a callable returning that vector at time `t`.
        imposed_vals : Callable[[float], jax.Array]
            Callable returning the full vector of imposed positions at time `t`.

        Returns
        -------
        jax.Array, shape: (2*nodes,)
            Internal reaction force on **all position DOFs** (no velocities), matching
            the sign used in the equations of motion:
                accel = (f_ext + f_internal - damping * xdot_free) / mass
        """
        # 1) Rebuild full x vector and reshape
        fixed_vals_t = fixed_vals(t) if callable(fixed_vals) else fixed_vals
        imposed_vals_t = imposed_vals(t)  # callable by construction above
        x_full = helpers_builders._assemble_full_from_free(free_mask, fixed_mask, imposed_mask, x_free, fixed_vals_t,
                                                           imposed_vals_t)  # full node positions, shape (2*nodes)
        pos_arr = helpers_builders._reshape_state_2_pos_arr(x_full, self.init_pos)

        # 2) Hinge torques: tau_hinges (H,)
        thetas = jax.vmap(lambda h: Strctr._get_theta(pos_arr, h))(jnp.arange(Strctr.hinges))  # (H,)
        B = self.buckle_arr  # (H,S)
        theta_eff = B * thetas[:, None]   # (H,S)
        # torque per shim
        tau_shims = - Variabs.torque(theta_eff)  # (H,S)
        # signed + summed per hinge
        tau_hinges = jnp.sum(B * tau_shims, axis=1)

        # 3) Jacobian of theta for each hinge: (H, n_coords)
        # --- efficient local-6DOF Jacobians for hinge angles ---
        def theta_of_x(x_flat, h):
            """θ_h(x_flat): scalar hinge angle for hinge h."""
            pa = helpers_builders._reshape_state_2_pos_arr(x_flat, self.init_pos)
            return Strctr._get_theta(pa, h)

        def hinge_local_dof_indices(h: int) -> jax.Array:
            """
            For hinge h in a chain, the angle depends on nodes (h, h+1, h+2),
            each with x,y → 6 position DOFs total.
            Returns a 1D array of length 6 with the global DOF indices.
            """
            # nodes: [h, h+1, h+2]
            nodes = jnp.array([h, h + 1, h + 2], dtype=jnp.int32)
            # DOFs: [2*node, 2*node+1] for each node
            dofs = jnp.stack([2 * nodes, 2 * nodes + 1], axis=1)  # (3, 2)
            return dofs.reshape(-1)  # (6,)

        def hinge_grad_global(x_flat, h: int) -> jax.Array:
            """
            Compute ∂θ_h/∂x_flat as a length-n_coords vector, but only using
            JVPs on the 6 local DOFs for hinge h.
            """
            local_idx = hinge_local_dof_indices(h)  # (6,)

            def f(x):
                return theta_of_x(x, h)

            def jvp_single(idx):
                v = jnp.zeros_like(x_flat)
                v = v.at[idx].set(1.0)
                _, g = jax.jvp(f, (x_flat,), (v,))
                return g  # scalar derivative dθ_h/dx_idx

            # derivatives with respect to the 6 local DOFs: (6,)
            g_local = jax.vmap(jvp_single)(local_idx)

            # scatter into global gradient vector (n_coords,)
            grad_global = jnp.zeros_like(x_flat)
            grad_global = grad_global.at[local_idx].set(g_local)
            return grad_global

        # theta_jacs: shape (H, n_coords)
        theta_jacs = jax.vmap(
            hinge_grad_global,
            in_axes=(None, 0),   # x_full shared, h varies
        )(x_full, jnp.arange(Strctr.hinges))

        # # old 3) Jacobian of theta for each hinge: (H, n_coords)
        # def theta_jac_of_h(h):
        #     def theta_of_x(x_flat):
        #         pa = x_flat.reshape(pos_arr.shape)
        #         return Strctr._get_theta(pa, h)
        #     return jax.jacrev(theta_of_x)(x_full[:Strctr.n_coords])  # (n_coords,)
        # theta_jacs = jax.vmap(theta_jac_of_h)(jnp.arange(Strctr.hinges))  # (H, n_coords)

        # 4) Map torques to DOF forces
        # F_theta_full = (theta_jacs.T @ tau_hinges).reshape(-1) / Strctr.L  # (n_coords,)
        F_theta_full = (theta_jacs.T @ tau_hinges).reshape(-1)  # (n_coords,)
        # 5) Edge stretch forces
        F_stretch_full = self.stretch_forces(Strctr, Variabs, pos_arr)  # (n_coords,)
        # Combine internal forces (reaction)
        F_internal_full = F_theta_full + F_stretch_full  # (n_coords,)

        # jax.debug.print('F_theta_full {}', F_theta_full)
        # jax.debug.print('F_stretch_full {}', F_stretch_full)

        return F_internal_full

    @eqx.filter_jit
    def total_potential_force_from_full_x(self, Variabs: "VariablesClass", Strctr: "StructureClass",
                                          x_full: jax.Array):
        """
        USED from rhs
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
        # 1) ------ pos_arr from x_full ------
        # already built x_full: (n_coords,) flattened positions ONLY (no velocities)
        pos_arr = helpers_builders._reshape_state_2_pos_arr(x_full, self.init_pos)

        # 2) ------ hinge torques per hinge (H,) ------
        thetas = jax.vmap(lambda h: Strctr._get_theta(pos_arr, h))(jnp.arange(Strctr.hinges))  # (H,)
        B = self.buckle_arr  # (H,S)
        theta_eff = B * thetas[:, None]  # (H,S)
        tau_shims = -Variabs.torque(theta_eff)  # (H,S)
        tau_hinges = jnp.sum(B * tau_shims, axis=1)  # (H,)

        # 3) ------ Jacobian of theta for each hinge: (H, n_coords) ------
        # --- efficient local-6DOF Jacobians for hinge angles ---
        def theta_of_x(x_flat, h):
            """θ_h(x_flat): scalar hinge angle for hinge h."""
            pa = helpers_builders._reshape_state_2_pos_arr(x_flat, self.init_pos)
            return Strctr._get_theta(pa, h)

        def hinge_local_dof_indices(h: int) -> jax.Array:
            """
            For hinge h in a chain, the angle depends on nodes (h, h+1, h+2),
            each with x,y → 6 position DOFs total.
            Returns a 1D array of length 6 with the global DOF indices.
            """
            # nodes: [h, h+1, h+2]
            nodes = jnp.array([h, h + 1, h + 2], dtype=jnp.int32)
            # DOFs: [2*node, 2*node+1] for each node
            dofs = jnp.stack([2 * nodes, 2 * nodes + 1], axis=1)  # (3, 2)
            return dofs.reshape(-1)  # (6,)

        def hinge_grad_global(x_flat, h: int) -> jax.Array:
            """
            Compute ∂θ_h/∂x_flat as a length-n_coords vector, but only using
            JVPs on the 6 local DOFs for hinge h.
            """
            local_idx = hinge_local_dof_indices(h)  # (6,)

            def f(x):
                return theta_of_x(x, h)

            def jvp_single(idx):
                v = jnp.zeros_like(x_flat)
                v = v.at[idx].set(1.0)
                _, g = jax.jvp(f, (x_flat,), (v,))
                return g  # scalar derivative dθ_h/dx_idx

            # derivatives with respect to the 6 local DOFs: (6,)
            g_local = jax.vmap(jvp_single)(local_idx)

            # scatter into global gradient vector (n_coords,)
            grad_global = jnp.zeros_like(x_flat)
            grad_global = grad_global.at[local_idx].set(g_local)
            return grad_global

        # theta_jacs: shape (H, n_coords)
        theta_jacs = jax.vmap(
            hinge_grad_global,
            in_axes=(None, 0),   # x_full shared, h varies
        )(x_full, jnp.arange(Strctr.hinges))

        # 3) ------ dense Jacobian (simple, OK for plotting; for scale use local-8DOF approach) ------
        # def theta_jac_of_h(h):
        #     def theta_of_x(x_flat):
        #         pa = helpers_builders._reshape_state_2_pos_arr(x_flat, self.init_pos)
        #         return Strctr._get_theta(pa, h)
        #     return jax.jacrev(theta_of_x)(x_full)  # (n_coords,)
        # theta_jacs = jax.vmap(theta_jac_of_h)(jnp.arange(Strctr.hinges))  # (H, n_coords)
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
        USED
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
            return jax.grad(lambda xf: -self.total_potential_energy_free(Variabs, Strctr, t, xf, free_mask=free_mask,
                                                                         fixed_mask=fixed_mask, imposed_mask=imposed_mask,
                                                                         fixed_vals=fixed_vals,
                                                                         imposed_vals=imposed_vals))(x_free)
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
    def force_function_free(self, t: float, force_function: Callable[[float], jax.Array], *, 
                            free_mask: jax.Array) -> jax.Array:
        """External force restricted to free DOFs."""
        return force_function(t)[free_mask]

    def solve_dynamics(self, state_0: jax.Array, Variabs: "VariablesClass", Strctr: "StructureClass",
                       fixed_mask: jax.Array[bool] = None, fixed_vals: jax.Array[jnp.float_] = None,
                       imposed_mask: jax.Array[bool] = None, imposed_vals: jax.Array[jnp.float_] = None, rtol: float = 1e-2,
                       maxsteps: int = 100):
        # ------ ensure correct sizes ---
        force_function = lambda t: jnp.zeros_like(self.init_pos).flatten()

        if fixed_mask is None:
            fixed_mask = jnp.zeros_like(self.init_pos).flatten().astype(bool)
        else:
            fixed_mask = jnp.array(fixed_mask).flatten().astype(bool)

        if fixed_vals is None:
            fixed_vals = jnp.asarray(self.init_pos, dtype=self.init_pos.dtype).reshape((self.init_pos.size,))
        elif callable(fixed_vals):
            # leave as-is
            pass
        else:
            # fixed_vals = jnp.asarray(fixed_vals, dtype=Eq.init_pos.dtype).reshape((Eq.init_pos.size,))
            ivec = jnp.asarray(fixed_vals, dtype=self.init_pos.dtype).reshape((self.init_pos.size,))
            fixed_vals = lambda t, ivec=ivec: ivec

        if imposed_mask is None:
            imposed_mask = jnp.zeros((self.init_pos.size,), dtype=bool)
        else:
            imposed_mask = jnp.asarray(imposed_mask, dtype=bool).reshape((self.init_pos.size,))

        # imposed displacement values: callable or constant vector
        if imposed_vals is None:
            base = jnp.asarray(self.init_pos, dtype=self.init_pos.dtype).reshape((self.init_pos.size,))
            imposed_vals = lambda t, base=base: base
        elif callable(imposed_vals):  # leave as-is
            pass
        else:
            ivec = jnp.asarray(imposed_vals, dtype=self.init_pos.dtype).reshape((self.init_pos.size,))
            imposed_vals = lambda t, ivec=ivec: ivec

        # # the speed at each DOF is just the derivative of the displacement, 
        # def compute_disp_speed(disp_func):
        #     # Returns another function that calculates derivative of displacement at time t, for jax.grad
        #     def disp_speed(t):
        #         # For each DOF where displacement is imposed, compute the derivative with respect to time
        #         # Use vmap to compute the derivative for all components efficiently
        #         # For each i, we create a function that gets the i-th component and compute its gradient
        #         component_grad = lambda i, t: jax.grad(lambda t_: disp_func(t_)[i])(t)
        #         # Apply this function to all indices using vmap Since imposed_vals is a function t -> vector
        #         full_derivative = jax.vmap(lambda i: component_grad(i, t))(jnp.arange(len(disp_func(t))))
        #         return full_derivative
        #     return disp_speed

        # # Create the speed function by applying the derivative computation
        # imposed_disp_speed_values = compute_disp_speed(imposed_vals)
        # jax.debug.print('imposed_disp_speed_values={}', imposed_disp_speed_values)
        # # imposed_disp_speed_values = grad(imposed_vals)

        free_mask, n_free_DOFs, state_0_free = helpers_builders._get_state_free_from_full(state_0, fixed_mask, imposed_mask)

        # ------ pure force function ------
        if self.calc_through_energy:
            force_full_fn = eqx.filter_jit(
                lambda x_full: jax.grad(lambda xf: -self.total_potential_energy(Variabs, Strctr, xf))(x_full))
        else:  # directly through forces, this is actually what's being used
            force_full_fn = eqx.filter_jit(
                lambda x_full: self.total_potential_force_from_full_x(Variabs, Strctr, x_full))

        # ------ right-hand-size of ODE ------
        @jit
        def rhs(state_free: jax.Array, t: float):
            x_free, xdot_free = state_free[:n_free_DOFs], state_free[n_free_DOFs:]
            f_ext = self.force_function_free(t, force_function, free_mask=free_mask)
            f_pot = self.potential_force_free(Variabs, Strctr, t, x_free, free_mask=free_mask, fixed_mask=fixed_mask,
                                              fixed_vals=fixed_vals, imposed_mask=imposed_mask, imposed_vals=imposed_vals)
            accel = (f_ext + f_pot - self.damping_coeff * xdot_free) / self.mass
            return jnp.concatenate([xdot_free, accel], axis=0)

        # @jit
        # def rhs_diffrax(t: float, state_free: jax.Array, args):
        #     return rhs(state_free, t)

        t1 = time.time()

        res_free: jax.Array = odeint(rhs, state_0_free, self.time_points, rtol=self.tolerance, mxstep=maxsteps)

        pos_mask = free_mask
        vel_mask = free_mask
        mask_free_both = jnp.concatenate([pos_mask, vel_mask], axis=0)

        res = jnp.zeros((res_free.shape[0], Strctr.n_coords * 2), dtype=res_free.dtype)
        res = res.at[:, mask_free_both].set(res_free)

        mask_fixed_pos = jnp.concatenate([fixed_mask, jnp.zeros_like(fixed_mask)], axis=0)
        mask_fixed_vel = jnp.concatenate([jnp.zeros_like(fixed_mask), fixed_mask], axis=0)
        res = res.at[:, mask_fixed_pos].set(vmap(fixed_vals)(self.time_points)[:, fixed_mask])
        res = res.at[:, mask_fixed_vel].set(0.0)

        mask_imposed_pos = jnp.concatenate([imposed_mask, jnp.zeros_like(imposed_mask)], axis=0)
        # mask_imposed_vel = jnp.concatenate([jnp.zeros_like(imposed_mask), imposed_mask], axis=0)

        res = res.at[:, mask_imposed_pos].set(vmap(imposed_vals)(self.time_points)[:, imposed_mask])
        # res = res.at[:, mask_imposed_vel].set(vmap(imposed_disp_speed_values)(self.time_points)[:, imposed_mask])

        # res.block_until_ready()
        print(f"Integration done in {time.time() - t1:.2f} s")

        final_disp = res[-1, :Strctr.n_coords].reshape(self.init_pos.shape)

        displacements = res[:, :Strctr.n_coords].reshape((len(res), self.init_pos.shape[0], self.init_pos.shape[1]))
        # .block_until_ready()

        # Get velocities from the last part of the state vector
        velocities = res[:, Strctr.n_coords:].reshape((len(res), self.init_pos.shape[0], self.init_pos.shape[1]))

        # we put the - to have the force as a reaction force
        potential_F_evolution = vmap(lambda x: force_full_fn(x[:Strctr.n_coords].reshape(self.init_pos.shape).flatten()))(res)
        # potential_force_evolution.block_until_ready()

        return (final_disp, displacements, velocities, potential_F_evolution)


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
