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
# Dynamics - Solve shape under displacement forcing
# ===================================================


def solve_dynamics(
    self,
    time_points,
    state_0: jax.Array,
    Variabs: "VariablesClass",
    Strctr: "StructureClass",
    force_function: jax.Array[jnp.float_] = None,
    fixed_DOFs: jax.Array[bool] = None,
    imposed_disp_DOFs: jax.Array[bool] = None,
    imposed_disp_values: jax.Array[jnp.float_] = None,  # function of time
    # simulation parameters
    damping: float = 0.01,
    rtol: float = 1e-2,
    maxsteps: int = 100,
):
    # print(state_0.shape)
    if force_function is None:
        force_function = lambda t: jnp.zeros_like(self.pos_arr).flatten()

    if fixed_DOFs is None:
        fixed_DOFs = jnp.zeros_like(self.pos_arr).flatten().astype(bool)
    else:
        fixed_DOFs = jnp.array(fixed_DOFs).flatten().astype(bool)

    if imposed_disp_DOFs is None:
        imposed_disp_DOFs = (
            jnp.zeros_like(self.pos_arr).flatten().astype(bool)
        )
        imposed_disp_values = lambda t: jnp.zeros_like(self.pos_arr).flatten()
        imposed_disp_speed_values = lambda t: jnp.zeros_like(self.pos_arr).flatten()
    else:
        imposed_disp_DOFs = jnp.array(imposed_disp_DOFs).flatten().astype(bool)

        # the speed at each DOF is just the derivative of the displacement
        # Create a function to compute displacement speed (derivative) using jax.grad
        # Since imposed_disp_values is a function t -> vector, we need to use vmap to apply grad to each component
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
        imposed_disp_speed_values = compute_disp_speed(imposed_disp_values)
        # imposed_disp_speed_values = grad(imposed_disp_values)
        # WIP, adapt to a function the size of the origami pts

    free_DOFs = jnp.logical_not(imposed_disp_DOFs | fixed_DOFs)

    n_free_DOFs = jnp.sum(free_DOFs)

    n_nodes = self.pos_arr.size

    def total_potential_energy_free(Variabs: "VariablesClass", Strctr: "StructureClass", t: float, x_free: jax.Array):
        """Total potential energy function that only returns the free DOFs."""
        x_full = jnp.zeros_like(self.pos_arr).flatten()
        x_full = x_full.at[free_DOFs].set(x_free)
        x_full = jnp.where(fixed_DOFs, 0.0, x_full)
        x_full = jnp.where(imposed_disp_DOFs, imposed_disp_values(t), x_full)
        return self.total_potential_energy(Variabs, Strctr, x_full)

    def potential_force_free_en(Variabs: "VariablesClass", Strctr: "StructureClass", t: float, x_free: jax.Array):
        # lambda t: grad(lambda x: -total_potential_energy_free(t, x))
        return grad(lambda x: -total_potential_energy_free(Variabs, Strctr, t, x))(x_free)

    potential_force = grad(lambda x: -self.total_potential_energy(Variabs, Strctr, x))

    def potential_force_free(t: float, x_free: jax.Array):
        # NOTE WIP Do this with the energy and not the force
        x_full = jnp.zeros_like(self.pos_arr).flatten()
        x_full = x_full.at[free_DOFs].set(x_free)
        x_full = jnp.where(fixed_DOFs, 0.0, x_full)
        x_full = jnp.where(imposed_disp_DOFs, imposed_disp_values(t), x_full)
        return potential_force(x_full)[free_DOFs]

    def force_function_free(t: float):
        """Force function that only returns the free DOFs."""
        return force_function(t)[free_DOFs]

    state_0 = state_0.flatten()
    state_0_x, state_0_x_dot = state_0[:n_nodes], state_0[n_nodes:]

    state_0_x_free = state_0_x[free_DOFs]
    state_0_x_dot_free = state_0_x_dot[free_DOFs]

    state_0_free = jnp.concatenate([state_0_x_free, state_0_x_dot_free])

    @jit
    def rhs(state_free: jax.Array, t: float):
        x_free, x_dot_free = state_free[:n_free_DOFs], state_free[n_free_DOFs:]
        accel = (force_function_free(t) + potential_force_free(t, x_free) - damping * x_dot_free) / 20.0
        return jnp.array([x_dot_free, accel], axis=0).flatten()

    @jit
    def rhs_diffrax(t: float, state_free: jax.Array, args):
        return rhs(state_free, t)

    t1 = time.time()

    res_free: jax.Array = odeint(
        rhs,
        state_0_free,
        time_points,
        rtol=rtol,
        mxstep=maxsteps,
    )

    # res_free = diffrax.diffeqsolve(
    #     terms=diffrax.ODETerm(rhs_diffrax),
    #     solver=diffrax.Dopri5(),
    #     t0=time_points.min(),
    #     t1=time_points.max(),
    #     # dt0=1e-1,  # None,
    #     dt0=None,  # None,
    #     y0=state_0_free,
    #     # args=args,
    #     stepsize_controller=diffrax.PIDController(
    #         rtol=1e-3, atol=1e-3
    #     ),  # , atol=atol, pcoeff=0., icoeff=1., dcoeff=0.),
    #     saveat=diffrax.SaveAt(ts=time_points),
    #     max_steps=None,
    #     adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=10),
    # ).ys

    pos_mask = free_DOFs
    vel_mask = free_DOFs
    mask_free_both = jnp.concatenate([pos_mask, vel_mask], axis=0)

    res = jnp.zeros((res_free.shape[0], n_nodes * 2), dtype=res_free.dtype)
    res = res.at[:, mask_free_both].set(res_free)

    mask_fixed_both = jnp.concatenate([fixed_DOFs, fixed_DOFs], axis=0)
    res = res.at[:, mask_fixed_both].set(0.0)

    mask_imposed_pos = jnp.concatenate([imposed_disp_DOFs, jnp.zeros_like(imposed_disp_DOFs)], axis=0)
    mask_imposed_vel = jnp.concatenate([jnp.zeros_like(imposed_disp_DOFs), imposed_disp_DOFs], axis=0)

    res = res.at[:, mask_imposed_pos].set(vmap(imposed_disp_values)(time_points)[:, imposed_disp_DOFs])
    res = res.at[:, mask_imposed_vel].set(vmap(imposed_disp_speed_values)(time_points)[:, imposed_disp_DOFs])

    # res = jnp.zeros((res_free.shape[0], n_nodes * 2))
    # res = res.at[:, jnp.concatenate([free_DOFs, free_DOFs])].set(res_free)
    # res = res.at[:, jnp.concatenate([fixed_DOFs, fixed_DOFs])].set(0.0)
    # res = res.at[
    #     :, jnp.concatenate([imposed_disp_DOFs, jnp.zeros_like(imposed_disp_DOFs)])
    # ].set(vmap(imposed_disp_values)(time_points)[:, imposed_disp_DOFs])
    # res = res.at[
    #     :, jnp.concatenate([jnp.zeros_like(imposed_disp_DOFs), imposed_disp_DOFs])
    # ].set(vmap(imposed_disp_speed_values)(time_points)[:, imposed_disp_DOFs])

    # res.block_until_ready()
    print(f"Integration done in {time.time() - t1:.2f} s")

    final_disp = res[-1, :n_nodes].reshape(self.pos_arr.shape)

    displacements = res[:, :n_nodes].reshape((len(res), self.pos_arr.shape[0], self.pos_arr.shape[1]))
    # .block_until_ready()

    # Get velocities from the last part of the state vector
    velocities = res[:, n_nodes:].reshape(
        (
            len(res),
            self.pos_arr.shape[0],
            self.pos_arr.shape[1],
        )
    )
    # .block_until_ready()

    # we put the - to have the force as a reaction force
    potential_force_evolution = vmap(
        lambda x: -potential_force(
            x[:n_nodes].reshape(self.initial_points.shape).flatten()
        )
    )(res)
    # potential_force_evolution.block_until_ready()

    return (
        final_disp,
        displacements,
        velocities,
        potential_force_evolution
    )
