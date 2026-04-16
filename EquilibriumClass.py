from __future__ import annotations

import copy
import jax
import jax.numpy as jnp
import equinox as eqx
from jax import grad, jit, vmap
from jax.experimental.ode import odeint

from typing import Tuple, List
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional

import helpers_builders

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

    Methods
    -------
    calculate_state(Variabs, Strctr, Sprvsr, init_pos, control_first_edge, tip_pos, tip_angle, pos_noise, vel_noise)
        Compute equilibrium state of the chain given boundary conditions and optional noise.
    total_potential_force(Variabs, Strctr, t, x_free, *, free_mask, fixed_mask, fixed_vals, imposed_mask, imposed_vals)
        Compute internal reaction force on all DOFs (positions only) given only x_free. 
        Fixed and imposed DOFs are reconstructed via the provided masks and callables.
    total_potential_force_from_full_x(Variabs, Strctr, x_full)
        Compute internal reaction force on all DOFs (positions only) given `x_full`.
    potential_force_free(Variabs, Strctr, t, x_free, *, free_mask, fixed_mask, fixed_vals, imposed_mask, imposed_vals)
        uses total_potential_force to get internal reaction force on **FREE** DOFs only, for the ODE RHS.
    bend_forces(Strctr, tau_hinges, x_full):
        Forces on all nodes due to torques on each hinge. to be assembled in total_...
    stretch_forces(Strctr, Variabs, jnp_pos_arr)
        Linear edge stretch forces on all DOFs given node positions. Should make sure distance between node is Strctr.L
    contact_forces_node_edge(Strctr, Variabs, jnp_pos_arr, edges, p, fmax, skip_band, eps)
        Compute node–edge contact forces (self-intersection prevention).
    force_function_free(t, force_function, free_mask)
        Restrict external forces to free DOFs only. Currently all zeros
    solve_dynamics(state_0, Variabs, Strctr, fixed_mask, fixed_vals, imposed_mask, imposed_vals, maxsteps)
        Integrate damped EOMs for chain on FREE DOFs, enforcing fixed and imposed DOFs through masks.

    Helpers:
    --------
    _theta_jacs_local(Strctr, x_flat)
        Compute local hinge-angle Jacobians ∂θ_h/∂x for all hinges h.
    _set_fixed_vals(fixed_mask)
        boolean jax array (2*N,), nonzero values are values that are fixed along equilibrium calculation 
        (i.e. 1st and 2nd nodes at chain base)
    _set_imposed_vals(Strctr, Sprvsr, tip_pos, tip_angle, init_pos):
        Build a time-dependent imposed displacement function.
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
    r_intersect_factor: float  # radius from which chain intersecting with itself produces repulsion, fraction of edge length
    k_intersect_factor: float  # force factor of repulsion due to chain intersecting with itself, later multiplied by k_stretch

    # ---- state / derived ----
    rest_lengths: jax.Array                          # (H+1,) edge rest lengths (from initial pos)
    jnp_init_pos: jax.Array                          # (hinges+2, 2) integer coordinates
    buckle_arr: jax.Array                            # (H,) ∈ {+1,-1} per hinge/shim (direction of stiff side)
    time_points: jax.Array                           # (T_eq, ) time steps for simulating equilibrium configuration
    
    def __init__(self, Strctr: "StructureClass", CFG: ExperimentConfig, ramp_pos: bool = True, buckle_arr: jax.Array = None,
                 pos_arr: jax.Array = None):
        self.damping_coeff = CFG.Eq.damping
        self.mass = CFG.Eq.mass        
        self.time_points = jnp.linspace(0, CFG.Eq.T_eq, int(60))
        self.tolerance = CFG.Eq.tolerance

        # default buckle: all +1
        if buckle_arr is None:
            self.buckle_arr = jnp.ones((Strctr.hinges, Strctr.shims), dtype=jnp.int32)
        else:
            self.buckle_arr = buckle_arr
            assert self.buckle_arr.shape == (Strctr.hinges, Strctr.shims)
            
        if pos_arr is None:
            self.jnp_init_pos = helpers_builders._initiate_pos(Strctr.edges+1, Strctr.L)  # (N=hinges+2, 2)
        else:
            self.jnp_init_pos = jnp.asarray(pos_arr)
            
        # each edge's rest length is L, it's fixed and very stiff 
        self.rest_lengths = jnp.full((Strctr.hinges + 1,), Strctr.L, dtype=jnp.float32)
        self.calc_through_energy = CFG.Eq.calc_through_energy
        self.ramp_pos = CFG.Eq.ramp_pos
        self.rand_key = CFG.Eq.rand_key_Eq
        self.pos_noise = CFG.Eq.pos_noise
        self.vel_noise = CFG.Eq.vel_noise
        self.r_intersect_factor = CFG.Eq.r_intersect_factor
        self.k_intersect_factor = CFG.Eq.k_intersect_factor

    # ---------------------------------------------------------------
    # main function of EquilibriumClass
    # ---------------------------------------------------------------
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
        forces: jax.Array, shape (N,)
            Time history of the internal reaction forces (stretch + bending) [mN]
            on each positional DOF, evaluated along the trajectory.

        Notes
        -----
        - The fixed and imposed DOFs are enforced directly inside `self.solve_dynamics`.
        - When `tip_angle` is imposed, the node before the tip is constrained so that
          the last segment has length `Strctr.L` and orientation `tip_angle`.
        - Positional noise is applied **after** flattening `init_pos` and **before**
          concatenating with the velocity vector.
        """
        if init_pos is None:
            # use whatever geometry was passed at construction as baseline
            init_pos = helpers_builders.jax2numpy(self.jnp_init_pos)

        jnp_init_pos = helpers_builders.numpy2jax(init_pos)

        # ------ fixed values (vector, not function) ------
        fixed_vals = self._set_fixed_vals(Strctr.fixed_mask)

        # ------ imposed tip position and possibly angle ------
        # Build a callable (always), even if mask is all False.
        imposed_vals = self._set_imposed_vals(Strctr, Sprvsr, tip_pos, tip_angle, jnp_init_pos)

        # -------- initial state (positions & velocities) ----------
        pos_noise = Strctr.L * self.pos_noise  # scale relative to length
        vel_noise = 1.5 * self.vel_noise  # from equating 1/2mv^2 = mean(Torque)*L
        state_0 = helpers_builders._extend_pos_to_x0_v0(jnp_init_pos, pos_noise, vel_noise, self.rand_key)

        # -------- run dynamics ----------
        final_pos, pos_in_t, vel_in_t, potential_F_in_t = self.solve_dynamics(state_0, Variabs, Strctr,
                                                                              fixed_mask=Strctr.fixed_mask, 
                                                                              fixed_vals=fixed_vals,
                                                                              imposed_mask=Sprvsr.imposed_mask,
                                                                              imposed_vals=imposed_vals)
        # print('STD forces=', jax.numpy.std(potential_F_in_t[300:], axis=0))
        # print('STD pos=', jax.numpy.std(pos_in_t[300:], axis=0))
        forces = potential_F_in_t[-1]  # [mN] from torque files

        return final_pos, pos_in_t, vel_in_t, forces

    # ---------------------------------------------------------------
    # assembly of total forces from bend, stretch, intersection
    # ---------------------------------------------------------------
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
          - Returns the full reaction force (size = n_coords, which is (2*nodes,)), from which the caller
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
            Internal reaction force on **all position DOFs** (no velocities) [mN], matching
            the sign used in the equations of motion:
                accel = (f_ext + f_internal - damping * xdot_free) / mass
        """
        # ------ Rebuild full x vector and reshape ------
        fixed_vals_t = fixed_vals(t) if callable(fixed_vals) else fixed_vals
        imposed_vals_t = imposed_vals(t)  # callable by construction above
        x_full = helpers_builders._assemble_full_from_free(free_mask, fixed_mask, imposed_mask, x_free, fixed_vals_t,
                                                           imposed_vals_t)  # full node positions, shape (2*nodes)
        jnp_pos_arr = helpers_builders._reshape_state_2_pos_arr(x_full, self.jnp_init_pos)

        # ------ Hinge torques: tau_hinges (H,) ------
        thetas = jax.vmap(lambda h: Strctr._get_theta(jnp_pos_arr, h))(jnp.arange(Strctr.hinges))  # (H,)
        B = self.buckle_arr  # (H,S), where S=1 is always incorporated here
        theta_eff = B * thetas[:, None]   # (H,S)
        # torque per shim
        tau_shims = - Variabs.torque(theta_eff)  # (H,S), experimental torque, exploding upon neighboring edge contact
        # signed + summed per hinge
        tau_hinges = jnp.sum(B * tau_shims, axis=1)  

        # Jacobian of theta for each hinge: (H, n_coords) which is (H, 2*nodes)
        theta_jacs = self._theta_jacs_local(Strctr, x_full)  # (H, n_coords)  

        # Map torques to DOF forces
        F_theta_full = (theta_jacs.T @ tau_hinges).reshape(-1)  # (n_coords,) which is (2*nodes,), [mN]
        
        # ------ Edge stretch forces ------
        F_stretch_full = self.stretch_forces(Strctr, Variabs, jnp_pos_arr)  # (n_coords,) which is (2*nodes,), [mN]

        # ------ Intersection of nodes and edges prohibited ------
        # not including contact of neighboring edges, this is strictly trough Variabs.torque
        F_contact_full = self.contact_forces_node_edge(Strctr, Variabs, jnp_pos_arr, Strctr.edges_arr, p=2.0, fmax=None,
                                                       skip_band=1)  # [mN]

        # jax.debug.print('F_theta_full {}', F_theta_full)
        # jax.debug.print('F_stretch_full {}', F_stretch_full)
        # jax.debug.print('F_contact_full {}', F_contact_full)
        # mag = lambda F: jnp.linalg.norm(F)

        # max_ext = jnp.max(mag(F_theta_full))
        # max_pot = jnp.max(mag(F_stretch_full))
        # max_con = jnp.max(mag(F_contact_full))

        # ------ Combine internal forces (reaction) ------
        # F_internal_full = F_theta_full + F_stretch_full  # (n_coords,) which is (2*nodes,)
        F_internal_full = F_theta_full + F_stretch_full + F_contact_full  # (n_coords,) which is (2*nodes,)

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
        x_full : jax.Array, shape: (n_coords,) which is (2*nodes,)
            Full position vector (no velocities).

        Returns
        -------
        jax.Array, shape: (n_coords,) which is (2*nodes,)
            Internal reaction force on **all position DOFs**. [mN] since torque is in [mN]
        """
        # ------ pos_arr from x_full ------
        # already built x_full: (n_coords,) flattened positions ONLY (no velocities)
        jnp_pos_arr = helpers_builders._reshape_state_2_pos_arr(x_full, self.jnp_init_pos)

        # ------ hinge torques per hinge (H,) ------
        thetas = jax.vmap(lambda h: Strctr._get_theta(jnp_pos_arr, h))(jnp.arange(Strctr.hinges))  # (H,)
        B = self.buckle_arr  # (H,S)
        theta_eff = B * thetas[:, None]  # (H,S)
        tau_shims = -Variabs.torque(theta_eff)  # (H,S)
        tau_hinges = jnp.sum(B * tau_shims, axis=1)  # (H,)

        # Map torques to DOF forces
        F_theta_full = self.bend_forces(Strctr, tau_hinges, x_full)

        # --- stretch forces ---
        F_stretch_full = self.stretch_forces(Strctr, Variabs, jnp_pos_arr)      # (n_coords,) which is (2*nodes,), [mN]

        F_contact_full = self.contact_forces_node_edge(Strctr, Variabs, jnp_pos_arr, Strctr.edges_arr, p=2.0, fmax=None,
                                                       skip_band=1)  # [mN]

        # jax.debug.print('F_theta_full {}', F_theta_full)
        # jax.debug.print('F_stretch_full {}', F_stretch_full)
        # jax.debug.print('F_contact_full {}', F_contact_full)

        return F_theta_full + F_stretch_full + F_contact_full                 # (n_coords,) which is (2*nodes,), [mN]

    def potential_force_free(self, Variabs: "VariablesClass", Strctr: "StructureClass", t: float,
                             x_free: jax.Array, *,
                             free_mask: jax.Array, fixed_mask: jax.Array, fixed_vals: jax.Array,
                             imposed_mask: jax.Array, imposed_vals: jax.Array):
        """
        uses total_potential_force to get internal reaction force on **FREE** DOFs only, for the ODE RHS.

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
        free_mask, fixed_mask, imposed_mask : jax.Array[bool], shape: (n_coords,) which is (2*nodes,)
        fixed_vals : Union[jax.Array, Callable[[float], jax.Array]]
        imposed_vals : Callable[[float], jax.Array]

        Returns
        -------
        jax.Array, shape: (sum(free_mask),)
            Internal reaction force on **free position DOFs** (restoring sign), [mN]
        """
        return self.total_potential_force(Variabs, Strctr, t, x_free, free_mask=free_mask, fixed_mask=fixed_mask,
                                          imposed_mask=imposed_mask, fixed_vals=fixed_vals, imposed_vals=imposed_vals)[free_mask]
 
    # ---------------------------------------------------------------
    # physical forces - bend, stretch, intersect
    # ---------------------------------------------------------------
    def bend_forces(self, Strctr, tau_hinges, x_full):
        """
        Forces on all nodes due to torques on each hinge.

        Returns:
        --------
        jax.Array, shape: (n_coords,) which is (2*nodes,), Flattened reaction force on all position DOFs, [mN]
        """

        theta_jacs = self._theta_jacs_local(Strctr, x_full)  # (H, n_coords)
        return (theta_jacs.T @ tau_hinges).reshape(-1)  # (n_coords,) which is (2*nodes,), [mN]  

    def stretch_forces(self, Strctr: "StructureClass", Variabs: "VariablesClass",
                       jnp_pos_arr: jax.Array[float]) -> jax.Array:
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
        jax.Array, shape: (n_coords,) which is (2*nodes,), Flattened reaction force on all position DOFs, [mN]
        """
        # pos_arr: (N, 2)
        # Strctr.edge_list: (E, 2) with node indices (ia, ib)
        # Strctr.rest_lengths: (E,)
        # Variabs.k_stretch: scalar or (E,)
        edges = Strctr.edges_arr             # (E,2) jax Array
        pa = jnp_pos_arr[edges[:, 0], :]          # (E,2)
        pb = jnp_pos_arr[edges[:, 1], :]          # (E,2)
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
        F = jnp.zeros_like(jnp_pos_arr)           # (N,2)
        F = F.at[edges[:, 0], :].add(+fvec)
        F = F.at[edges[:, 1], :].add(-fvec)
        return F.reshape(-1)                  # (n_coords,) which is (2*nodes,)

    def contact_forces_node_edge(self, Strctr: "StructureClass", Variabs: "VariablesClass", jnp_pos_arr: jax.Array, 
                                 edges: jax.Array, p: float = 1.0, fmax: float | None = None, skip_band: int = 1,
                                 eps: float = 1e-12):
        """
        Compute node–edge contact (self-intersection prevention) forces for a planar chain.
        Computes the shortest distance from node i to the edge (j,k). 
        If the node penetrates within a radius r = r_intersect_factor * L, a repulsive force is applied along the outward normal.
        Equal and opposite forces are distributed to the segment endpoints (j,k).
        To avoid fighting stretch constraints, edges whose endpoints are within `skip_band` from the node are ignored.

        Parameters:
        ------------
        jnp_pos_arr: (N,2) jax array, node positions in x-y
        edges      : (NE, 2) of each edge (1st dim) connecting node i to j (2nd dim)
        p          : exponent on penetration (1=linear, 2=quadratic)
        fmax       : float, maximal force that can be applied while using node-edge contact 
        skip_band  : skip edges within this index distance (chain), don't measure contact between a node and its own edge
        eps    

        Returns
        -------
        F_contact_full : jax.Array, shape (2*N,). Flattened contact force [mN] on all nodes, ordered as:
                         [Fx0, Fy0, Fx1, Fy1, ..., Fx_{N-1}, Fy_{N-1}] 
        """
        r = self.r_intersect_factor*Strctr.L
        k = self.k_intersect_factor*Variabs.k_stretch

        N = jnp_pos_arr.shape[0]

        def pair_force(i, e):
            j, kidx = e  # segment endpoints
            # For a plain chain (0-1-2-...): skip nearby edges to avoid fighting elasticity
            # Condition: |i-j| <= skip_band OR |i-k| <= skip_band
            near = (jnp.abs(i - j) <= skip_band) | (jnp.abs(i - kidx) <= skip_band)

            pnt = jnp_pos_arr[i]
            a = jnp_pos_arr[j]
            b = jnp_pos_arr[kidx]

            dvec, d2, t = helpers_builders._point_segment_closest(pnt, a, b, eps=eps)  # dvec = p-c
            d = jnp.sqrt(d2 + eps)

            # penetration depth
            delta = r - d
            active = (delta > 0.0) & (~near)

            # unit normal (repel direction)
            n = dvec / (d + eps)

            # force magnitude
            Fmag = k * (delta ** p)
            if fmax is not None:
                Fmag = jnp.minimum(Fmag, fmax)

            F = jnp.where(active, Fmag * n, jnp.zeros_like(n))  # force on node i, [mN]

            # Distribute equal and opposite on the segment endpoints
            Fa = -(1.0 - t) * F
            Fb = -t * F

            return F, Fa, Fb, j, kidx

        # Accumulate forces via scatter-add
        Ftot = jnp.zeros_like(jnp_pos_arr)

        ii = jnp.arange(N)

        # vmap over i and edges -> (N,M,...)  (OK for N~small/moderate; optimize later if needed)
        def for_i(i):
            return jax.vmap(lambda e: pair_force(i, e))(edges)

        F_i, Fa_i, Fb_i, j_i, k_i = jax.vmap(for_i)(ii)

        # Sum contributions:
        # Node forces: add over edges
        Ftot = Ftot.at[ii].add(jnp.sum(F_i, axis=1))  # (N,2), [mN]

        # Edge endpoint forces: need scatter add with indices j_i,k_i (shape N,M)
        # Flatten N,M -> NM for scatter
        j_flat = j_i.reshape(-1)
        k_flat = k_i.reshape(-1)
        Fa_flat = Fa_i.reshape(-1, 2)
        Fb_flat = Fb_i.reshape(-1, 2)

        Ftot = Ftot.at[j_flat].add(Fa_flat)
        Ftot = Ftot.at[k_flat].add(Fb_flat)

        return Ftot.reshape(-1)

    # -------- external forces ----------
    def force_function_free(self, t: float, force_function: Callable[[float], jax.Array], *,
                            free_mask: jax.Array) -> jax.Array:
        """
        External force restricted to free DOFs. returns all zeros as of 2026Mar
        """
        return force_function(t)[free_mask]

    # ---------------------------------------------------------------
    # ODE solve
    # ---------------------------------------------------------------
    def solve_dynamics(self, state_0: jax.Array, Variabs: "VariablesClass", Strctr: "StructureClass",
                       fixed_mask: jax.Array[bool] = None, fixed_vals: jax.Array[jnp.float64] = None,
                       imposed_mask: jax.Array[bool] = None, imposed_vals: jax.Array[jnp.float64] = None,
                       maxsteps: int = 1000):
        """
        Integrate damped EOMs for chain on FREE DOFs, enforcing fixed and imposed DOFs through masks.

        This method:
          1. Extracts the reduced state vector containing only FREE positional DOFs and their velocities.
              - fixed_vals / imposed_vals become callables: t -> full position vector (n_coords,)
          2. Integrates the ODE on the reduced state using `jax.experimental.ode.odeint`.
          3. Reassembles the full trajectory (positions+velocities) by inserting imposed and fixed
          4. Computes internal reaction forces along the trajectory.

        Parameters
        ----------
        state_0                  : jax.Array, shape (2*n_coords,)
                                   full state vector before equilibration, with positions and velocities, concatenated as:
                                   [x0, y0, x1, y1, ..., x_{N-1}, y_{N-1}, vx0, vy0, ..., vx_{N-1}, vy_{N-1}]
                                   where n_coords = 2*nodes.
        fixed_mask, imposed_mask : optional array of bool, (n_coords,). 
                                   True at DOFs that remain fixed / imposed throughout the integration.
        fixed_vals, imposed_vals : jax.Array or callable, optional
                                   If an array: full position vector of (n_coords,) containing fixed / imposed DOF values
                                   (other entries ignored). Converted to a callable internally.
                                   If a callable: function f(t) -> full position vector (n_coords,).
                                   If None: defaults to the initial positions stored in `self.jnp_init_pos`.
        maxsteps                 : int, default=100. Maximum number of internal steps for `odeint` (passed as `mxstep`).

        Returns
        -------
        final_pos        : jax.Array, (nodes, 2). Final nodal positions at end of the integration (positions only).
        pos_in_t         : jax.Array, (T, nodes, 2). Time history of nodal positions, reconstructed from full state history.
        vel_in_t         : jax.Array, (T, nodes, 2). Time history of nodal velocities.
        potential_F_in_t : jax.Array, shape (T, n_coords). Internal reaction forces on position DOFs (flattened).

        Notes
        -----
        - The ODE is solved only on free DOFs. Fixed and imposed DOFs are injected back
          into the full state after integration.
        - External forces are currently set to zero (`force_function` is identically zero).
        - Imposed DOFs are treated as prescribed positions; their velocities are not explicitly
          imposed here (kept at zero unless they are also free DOFs).
        """
        # ------ external forces (currently zero) ------
        force_function = lambda t: jnp.zeros_like(self.jnp_init_pos).reshape(-1)

        # ------ canonicalize masks ------
        if fixed_mask is None:
            fixed_mask = jnp.zeros_like(self.jnp_init_pos).reshape(-1).astype(bool)
        else:
            fixed_mask = jnp.asarray(fixed_mask).reshape(-1).astype(bool)

        if imposed_mask is None:
            imposed_mask = jnp.zeros((self.jnp_init_pos.size,), dtype=bool)
        else:
            imposed_mask = jnp.asarray(imposed_mask, dtype=bool).reshape((self.jnp_init_pos.size,))

        # ------ canonicalize fixed_vals to a callable ------
        if fixed_vals is None:
            ivec = jnp.asarray(self.jnp_init_pos, dtype=self.jnp_init_pos.dtype).reshape(-1)
            fixed_vals = lambda t, ivec=ivec: ivec
        elif callable(fixed_vals):
            pass
        else:
            ivec = jnp.asarray(fixed_vals, dtype=self.jnp_init_pos.dtype).reshape(-1)
            fixed_vals = lambda t, ivec=ivec: ivec

        # ------ canonicalize imposed_vals to a callable ------
        if imposed_vals is None:
            base = jnp.asarray(self.jnp_init_pos, dtype=self.jnp_init_pos.dtype).reshape(-1)
            imposed_vals = lambda t, base=base: base
        elif callable(imposed_vals):
            pass
        else:
            ivec = jnp.asarray(imposed_vals, dtype=self.jnp_init_pos.dtype).reshape(-1)
            imposed_vals = lambda t, ivec=ivec: ivec

        # ------ reduce full state to free DOFs ------
        free_mask, n_free_DOFs, state_0_free = helpers_builders._get_state_free_from_full(state_0, fixed_mask, imposed_mask)

        # JIT force-from-full-x function for fast force evaluations along the trajectory
        force_full_fn = eqx.filter_jit(lambda x_full: self.total_potential_force_from_full_x(Variabs, Strctr, x_full))

        # ------ RHS on the reduced (free) state ------
        @jit
        def rhs(state_free: jax.Array, t: float):
            x_free, xdot_free = state_free[:n_free_DOFs], state_free[n_free_DOFs:]

            f_ext = self.force_function_free(t, force_function, free_mask=free_mask)
            f_pot = self.potential_force_free(Variabs, Strctr, t, x_free, free_mask=free_mask, fixed_mask=fixed_mask,
                                              fixed_vals=fixed_vals, imposed_mask=imposed_mask, imposed_vals=imposed_vals)
            accel = (f_ext + f_pot - self.damping_coeff * xdot_free) / self.mass
            return jnp.concatenate([xdot_free, accel], axis=0)

        # ------ integrate reduced system ------
        res_free: jax.Array = odeint(rhs, state_0_free, self.time_points, rtol=self.tolerance, mxstep=maxsteps)

        # ------ reconstruct full state history (positions+velocities) ------
        # free DOFs occupy the same mask in both position and velocity halves
        mask_free_both = jnp.concatenate([free_mask, free_mask], axis=0)

        res = jnp.zeros((res_free.shape[0], Strctr.n_coords * 2), dtype=res_free.dtype)
        res = res.at[:, mask_free_both].set(res_free)

        # fixed positions injected; fixed velocities set to zero
        mask_fixed_pos = jnp.concatenate([fixed_mask, jnp.zeros_like(fixed_mask)], axis=0)
        mask_fixed_vel = jnp.concatenate([jnp.zeros_like(fixed_mask), fixed_mask], axis=0)
        res = res.at[:, mask_fixed_pos].set(vmap(fixed_vals)(self.time_points)[:, fixed_mask])
        res = res.at[:, mask_fixed_vel].set(0.0)

        # imposed positions injected; imposed velocities left as-is (currently zero unless free)
        mask_imposed_pos = jnp.concatenate([imposed_mask, jnp.zeros_like(imposed_mask)], axis=0)
        res = res.at[:, mask_imposed_pos].set(vmap(imposed_vals)(self.time_points)[:, imposed_mask])

        # ------ unpack outputs ------
        final_pos = res[-1, :Strctr.n_coords].reshape(self.jnp_init_pos.shape)

        pos_in_t = res[:, :Strctr.n_coords].reshape((len(res), self.jnp_init_pos.shape[0], self.jnp_init_pos.shape[1]))
        vel_in_t = res[:, Strctr.n_coords:].reshape((len(res), self.jnp_init_pos.shape[0], self.jnp_init_pos.shape[1]))

        # reaction/internal forces on positions along the trajectory
        potential_F_in_t = vmap(lambda x: force_full_fn(x[:Strctr.n_coords]))(res)

        return (final_pos, pos_in_t, vel_in_t, potential_F_in_t)

    # ---------------------------------------------------------------
    # Physical helpers
    # ---------------------------------------------------------------
    def _theta_jacs_local(self, Strctr: "StructureClass", x_flat: jax.Array) -> jax.Array:
        """
        Compute local hinge-angle Jacobians ∂θ_h/∂x for all hinges h.
        For a simple chain, hinge h depends only on the positions of nodes:
            (h), (h+1), (h+2),
        i.e. 3 nodes × 2 coordinates = 6 position DOFs.

        Parameters:
        -----------
        x_flat - jax.Array, shape (2*nodes,). Flattened **position** vector (no velocities), ordered as:
                                              [x0, y0, x1, y1, ..., x_{N-1}, y_{N-1}]

        Returns
        -------
        theta_jacs - jax.Array, shape (H, n_coords) which is (H, 2*nodes)
            Row h contains the gradient of hinge angle θ_h w.r.t. all position DOFs.
            Only the 6 DOFs of nodes (h, h+1, h+2) are non-zero.
        """
        H = int(Strctr.hinges)
        zero = jnp.zeros_like(x_flat)

        def local_idx_for_h(h: jax.Array) -> jax.Array:
            """
            Global DOF indices for nodes (h, h+1, h+2):
            [2h, 2h+1, 2(h+1), 2(h+1)+1, 2(h+2), 2(h+2)+1]
            """
            base = 2 * h
            return jnp.array([base, base + 1, base + 2, base + 3, base + 4, base + 5], dtype=jnp.int32)

        def grad_for_h(h: jax.Array) -> jax.Array:
            idx = local_idx_for_h(h)     # (6,)
            x0 = x_flat[idx]             # (6,)

            def theta_of_local(x_local: jax.Array) -> jax.Array:
                # only patch local 6 dofs into the full vector
                x_full = x_flat.at[idx].set(x_local)
                pos_arr = helpers_builders._reshape_state_2_pos_arr(x_full, self.jnp_init_pos)
                return Strctr._get_theta(pos_arr, h)

            g_local = jax.jacrev(theta_of_local)(x0)  # (6,)
            return zero.at[idx].set(g_local)          # (2*nodes,)

        return jax.vmap(grad_for_h)(jnp.arange(H, dtype=jnp.int32)) 

    # ---------------------------------------------------------------
    # Helpers for assembly - set vals
    # ---------------------------------------------------------------
    def _set_fixed_vals(self, fixed_mask) -> jax.array:
        """
        return jax array (2*N,), nonzero values are values that are fixed along equilibrium calculation 
        (i.e. 1st and 2nd nodes at chain base)

        Parameters:
        -----------
        fixed_mask   : jnp.array(bool), (2 * nodes,), which of the nodes (in x,y) are fixed.

        Returns:
        --------
        fixed_vals : jnp.ndarray, shape (2 * nodes,), dtype=bool, [x0, y0, x1, y1, ..., x_last, y_last]
        """
        fixed_vals = jnp.zeros((len(fixed_mask),), dtype=float)
        return fixed_vals.at[fixed_mask].set(self.jnp_init_pos.reshape((-1,))[fixed_mask])

    def _set_imposed_vals(self, Strctr: "StructureClass", Sprvsr: "SupervisorClass", tip_pos, tip_angle, 
                          init_pos):
        """
        Build a time-dependent imposed displacement function.

        - At t = 0: imposed DOFs equal the previous equilibrium positions (self.jnp_init_pos).
        - For 0 < t < T_ramp: linearly ramp to the new tip pose.
        - For t >= T_ramp: imposed DOFs stay at the new pose.

        Parameters:
        -----------
        Sprvsr.imposed_mask: jnp.array(bool) Boolean mask of shape (2 * nodes,) marking imposed DOFs.

        tip_pos : array-like or jax array, optional
            tip position at end of training step, shape (2,) = (x_tip, y_tip). If None, no imposed update is applied.

        tip_angle : float or jax array, optional,
            tip angle in at end of training step in radians (CCW from +x). If None, the initial tip angle
            (angle of the last edge in `init_pos`) is preserved during the ramp.

        init_pos : jax array, shape (nodes, 2)
            Initial nodal positions used as the starting configuration at t=0 and as reference for ramp.

        Returns:
        --------
        imposed_vals : Callable[[float], jax.Array]
            A function of time `t` returning flattened position vector (2 * nodes,). Imposed entries are set to the values:
                [x0, y0, x1, y1, ..., x_{N-1}, y_{N-1}]

            Used as `imposed_mask` (or `Sprvsr.imposed_mask`) in `solve_dynamics()`
        """
        start_vec = init_pos.reshape(-1)

        if tip_pos is None:
            return lambda t, v=start_vec: v

        tip_init = start_vec[-2:]  # only if last node is tip in your flattening
        # Better: explicitly pull from init_pos:
        tip_init = init_pos[-1, :]                      # (2,)
        before_tip_init = init_pos[-2, :]                   # (2,)
        # wrapped angle of last edge, unwrap later since theta_fin is known
        theta_init_wrapped = jnp.arctan2(tip_init[1]-before_tip_init[1], tip_init[0]-before_tip_init[0])

        tip_fin = jnp.asarray(tip_pos, dtype=init_pos.dtype).reshape((2,))
        theta_fin = jnp.asarray(tip_angle, dtype=init_pos.dtype) if tip_angle is not None else theta_init_wrapped

        # Unwrap theta_init so it is the equivalent angle closest to theta_fin
        two_pi = jnp.asarray(2.0 * jnp.pi, dtype=init_pos.dtype)
        theta_init = theta_init_wrapped + two_pi * jnp.round((theta_fin - theta_init_wrapped) / two_pi)

        T_total = self.time_points[-1]
        T_ramp = 0.33 * T_total

        def smoothstep(s):
            return s*s*(3 - 2*s)  # C1 smooth

        def imposed_vals(t):
            s = jnp.clip(t / T_ramp, 0.0, 1.0)
            s = smoothstep(s)

            tip_t = (1-s) * tip_init + s * tip_fin
            th_t = (1-s) * theta_init + s * theta_fin

            before_t = helpers_builders._get_before_tip(
                tip_pos=tip_t, tip_angle=th_t, L=Strctr.L, dtype=init_pos.dtype
            )
            out = start_vec
            out = out.at[Sprvsr.imposed_mask].set(jnp.concatenate([before_t, tip_t]))
            return out

        return imposed_vals

# # ==========
# # NOT IN USE
# # ==========

# def calculate_energy_in_t(self, Variabs: "VariablesClass", Strctr: "StructureClass",
#                           displacements: NDArray[np.float64]) -> jax.array:
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
