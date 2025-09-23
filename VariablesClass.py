from __future__ import annotations
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from typing import Tuple, List
from numpy import array, zeros
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional

if TYPE_CHECKING:
    from StructureClass import StructureClass

import file_funcs

# ===================================================
# Class - User Variables: stiffnesses, lengths, etc.
# ===================================================


class VariablesClass(eqx.Module):
    # Make it static so Python control flow on it is outside JIT.
    k_type: str = eqx.field(static=True)

    k_soft: jax.Array | None = eqx.field(default=None)
    k_stiff: jax.Array | None = eqx.field(default=None)

    # k is a callable when using experimental data
    k: callable | None = eqx.field(default=None)

    thetas_ss: jax.Array = eqx.field(init=False)
    thresh: jax.Array = eqx.field(init=False)
    k_stretch: jax.Array = eqx.field(init=False)

    def __init__(self,
                 Strctr: "StructureClass",
                 k_type: str = "Numerical",
                 k_soft: jax.Array | None = None,
                 k_stiff: jax.Array | None = None,
                 thetas_ss: jax.Array | None = None,
                 thresh: jax.Array | None = None,
                 stretch_scale: float = 1e3,
                 file_name: str | None = None):
        H, S = Strctr.hinges, Strctr.shims
        self.k_type = k_type  # static

        if k_type == "Numerical":
            self.k_soft  = jnp.ones((H, S), jnp.float32) if k_soft  is None else jnp.asarray(k_soft,  jnp.float32)
            self.k_stiff = jnp.ones((H, S), jnp.float32) if k_stiff is None else jnp.asarray(k_stiff, jnp.float32)
            self.k = None
            # protect against None
            kmax = jnp.max(self.k_stiff)
            self.k_stretch = jnp.asarray(stretch_scale * kmax, jnp.float32)

        elif k_type == "Experimental":
            self.k_soft = None
            self.k_stiff = None
            thetas, torques, ks, torque_of_theta, k_of_theta = file_funcs.build_torque_stiffness_from_file(
                file_name, savgol_window=9
            )
            # store the callable that maps theta -> k(theta)
            self.k = k_of_theta
            self.k_stretch = jnp.asarray(stretch_scale * jnp.max(ks), jnp.float32)

        self.thetas_ss = (jnp.ones((H, S), jnp.float32) if thetas_ss is None
                          else jnp.asarray(thetas_ss, jnp.float32))
        self.thresh = (jnp.ones((H, S), jnp.float32) if thresh is None
                       else jnp.asarray(thresh, jnp.float32))

        
        
