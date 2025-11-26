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
    from EqClass import EqClass

import helpers_builders


# ===================================================
# Dynamics - Solve shape under displacement forcing
# ===================================================


def solve_dynamics(
    state_0: jax.Array,
    Variabs: "VariablesClass",
    Strctr: "StructureClass",
    Eq: "EqClass",
    force_function: jax.Array[jnp.float_] = None,
    fixed_DOFs: jax.Array[bool] = None,
    fixed_vals: jax.Array[jnp.float_] = None,
    imposed_DOFs: jax.Array[bool] = None,
    imposed_vals: jax.Array[jnp.float_] = None,  # function of time
    # simulation parameters
    rtol: float = 1e-2,
    maxsteps: int = 100,
):

    # -------- time grid ----------
    time_points = Eq.time_points
    damping = Eq.damping_coeff
    mass = Eq.mass
    tolerance = Eq.tolerance

    # print(state_0.shape)
    if force_function is None:
        force_function = lambda t: jnp.zeros_like(Eq.init_pos).flatten()

    if fixed_DOFs is None:
        fixed_DOFs = jnp.zeros_like(Eq.init_pos).flatten().astype(bool)
    else:
        fixed_DOFs = jnp.array(fixed_DOFs).flatten().astype(bool)

    # if fixed_vals is None:
    #     fixed_vals = jnp.zeros_like((fixed_DOFs,), dtype=Eq.init_pos.dtype)
    # else:
    #     fixed_vals = jnp.asarray(fixed_vals, dtype=Eq.init_pos.dtype)

    if fixed_vals is None:
        fixed_vals = jnp.asarray(Eq.init_pos, dtype=Eq.init_pos.dtype).reshape((Eq.init_pos.size,))
    elif callable(fixed_vals):
        # leave as-is
        pass
    else:
        # fixed_vals = jnp.asarray(fixed_vals, dtype=Eq.init_pos.dtype).reshape((Eq.init_pos.size,))
        ivec = jnp.asarray(fixed_vals, dtype=Eq.init_pos.dtype).reshape((Eq.init_pos.size,))
        fixed_vals = lambda t, ivec=ivec: ivec

    if imposed_DOFs is None:
        imposed_DOFs = jnp.zeros((Eq.init_pos.size,), dtype=bool)
    else:
        imposed_DOFs = jnp.asarray(imposed_DOFs, dtype=bool).reshape((Eq.init_pos.size,))

    # --- imposed displacement values: callable or constant vector ---
    if imposed_vals is None:
        base = jnp.asarray(Eq.init_pos, dtype=Eq.init_pos.dtype).reshape((Eq.init_pos.size,))
        imposed_vals = lambda t, base=base: base
    elif callable(imposed_vals):
        # leave as-is
        pass
    else:
        ivec = jnp.asarray(imposed_vals, dtype=Eq.init_pos.dtype).reshape((Eq.init_pos.size,))
        imposed_vals = lambda t, ivec=ivec: ivec

    # imposed_disp_speed_values = lambda t: jnp.zeros_like(imposed_DOFs).flatten()

    # the speed at each DOF is just the derivative of the displacement
    # Create a function to compute displacement speed (derivative) using jax.grad
    # Since imposed_vals is a function t -> vector, we need to use vmap to apply grad to each component
    def compute_disp_speed(disp_func):
        # This function returns another function that calculates the derivative of displacement at time t
        def disp_speed(t):
            # For each DOF where displacement is imposed, compute the derivative with respect to time
            # Use vmap to compute the derivative for all components efficiently
            # For each i, we create a function that gets the i-th component and compute its gradient
            component_grad = lambda i, t: jax.grad(lambda t_: disp_func(t_)[i])(t)
            # Apply this function to all indices using vmap
            full_derivative = jax.vmap(lambda i: component_grad(i, t))(jnp.arange(len(disp_func(t))))
            return full_derivative
        return disp_speed

    # Create the speed function by applying the derivative computation
    imposed_disp_speed_values = compute_disp_speed(imposed_vals)
    jax.debug.print('imposed_disp_speed_values=', imposed_disp_speed_values)
    # imposed_disp_speed_values = grad(imposed_vals)
    # WIP, adapt to a function the size of the origami pts

    free_DOFs = jnp.logical_not(imposed_DOFs | fixed_DOFs)
    n_free_DOFs = jnp.sum(free_DOFs)

    state_0 = state_0.flatten()
    state_0_x, state_0_x_dot = state_0[:Strctr.n_coords], state_0[Strctr.n_coords:]
    state_0_x_free = state_0_x[free_DOFs]
    state_0_x_dot_free = state_0_x_dot[free_DOFs]
    state_0_free = jnp.concatenate([state_0_x_free, state_0_x_dot_free])

    # A pure function of x_full -> (n_coords,) with reaction sign.
    if Eq.calc_through_energy:
        force_full_fn = eqx.filter_jit(
            lambda x_full: jax.grad(lambda xf: -Eq.total_potential_energy(Variabs, Strctr, xf))(x_full)
        )
    else:
        force_full_fn = eqx.filter_jit(
            lambda x_full: Eq.total_potential_force_from_full_x(Variabs, Strctr, x_full)
        )

    @jit
    def rhs(state_free: jax.Array, t: float):
        x_free, xdot_free = state_free[:n_free_DOFs], state_free[n_free_DOFs:]
        f_ext = Eq.force_function_free(t, force_function, free_mask=free_DOFs)
        f_pot = Eq.potential_force_free(
            Variabs, Strctr, t, x_free,
            free_mask=free_DOFs,
            fixed_mask=fixed_DOFs,
            fixed_vals=fixed_vals,
            imposed_mask=imposed_DOFs,
            imposed_vals=imposed_vals
        )
        # jax.debug.print('f_pot={}', f_pot)
        # jax.debug.print('f_ext={}', f_ext)
        # jax.debug.print('damping * xdot_free={}', damping * xdot_free)
        accel = (f_ext + f_pot - damping * xdot_free) / mass
        # jax.debug.print('accel={}', accel)
        return jnp.concatenate([xdot_free, accel], axis=0)

    @jit
    def rhs_diffrax(t: float, state_free: jax.Array, args):
        return rhs(state_free, t)

    t1 = time.time()

    res_free: jax.Array = odeint(
        rhs,
        state_0_free,
        time_points,
        rtol=tolerance,
        mxstep=maxsteps,
    )

    pos_mask = free_DOFs
    vel_mask = free_DOFs
    mask_free_both = jnp.concatenate([pos_mask, vel_mask], axis=0)

    res = jnp.zeros((res_free.shape[0], Strctr.n_coords * 2), dtype=res_free.dtype)
    res = res.at[:, mask_free_both].set(res_free)

    mask_fixed_pos = jnp.concatenate([fixed_DOFs, jnp.zeros_like(fixed_DOFs)], axis=0)
    mask_fixed_vel = jnp.concatenate([jnp.zeros_like(fixed_DOFs), fixed_DOFs], axis=0)
    res = res.at[:, mask_fixed_pos].set(vmap(fixed_vals)(time_points)[:, fixed_DOFs])
    res = res.at[:, mask_fixed_vel].set(0.0)

    mask_imposed_pos = jnp.concatenate([imposed_DOFs, jnp.zeros_like(imposed_DOFs)], axis=0)
    mask_imposed_vel = jnp.concatenate([jnp.zeros_like(imposed_DOFs), imposed_DOFs], axis=0)

    res = res.at[:, mask_imposed_pos].set(vmap(imposed_vals)(time_points)[:, imposed_DOFs])
    res = res.at[:, mask_imposed_vel].set(vmap(imposed_disp_speed_values)(time_points)[:, imposed_DOFs])

    # res.block_until_ready()
    print(f"Integration done in {time.time() - t1:.2f} s")

    final_disp = res[-1, :Strctr.n_coords].reshape(Eq.init_pos.shape)

    displacements = res[:, :Strctr.n_coords].reshape((len(res), Eq.init_pos.shape[0], Eq.init_pos.shape[1]))
    # .block_until_ready()

    # Get velocities from the last part of the state vector
    velocities = res[:, Strctr.n_coords:].reshape(
        (
            len(res),
            Eq.init_pos.shape[0],
            Eq.init_pos.shape[1],
        )
    )
    # .block_until_ready()

    # x_full = helpers_builders._assemble_full(free_DOFs, fixed_DOFs, imposed_DOFs, res_free,
    #                                          fixed_vals(time_points[0]), imposed_vals(time_points[0]))
    # for i, x_full in enumerate(res[:, :Strctr.n_coords]):
    #     err_dict = Eq.check_force_sign(Variabs, Strctr, x_full)
    #     jax.debug.print('cos sim {}', err_dict["cosine_similarity"])
    #     err_dict_free = Eq.check_force_sign(Variabs, Strctr, x_full, free_mask=pos_mask)
    #     jax.debug.print('cos sim free {}', err_dict_free["cosine_similarity"])

    # we put the - to have the force as a reaction force
    potential_force_evolution = vmap(
        lambda x: force_full_fn(
            x[:Strctr.n_coords].reshape(Eq.init_pos.shape).flatten()
        )
    )(res)
    # potential_force_evolution.block_until_ready()

    return (
        final_disp,
        displacements,
        velocities,
        potential_force_evolution
    )
