from __future__ import annotations
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from typing import Tuple, List
from numpy import array, zeros
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from StructureClass import StructureClass

import file_funcs

# ===================================================
# Class - User Variables: stiffnesses, lengths, etc.
# ===================================================


class VariablesClass(eqx.Module):
    # --- configuration / mode ---
    k_type: str = eqx.field(static=True)  # "Numerical" | "Experimental"

    # if using numerical values for springs
    k_soft: Optional[jax.Array] = eqx.field(default=None, static=True)   # (H, S)
    k_stiff: Optional[jax.Array] = eqx.field(default=None, static=True)  # (H, S)

    # k is a callable when using experimental data
    k: Optional[Callable[[jax.Array], jax.Array]] = eqx.field(default=None, static=True)

    # maximal spring constant for normalizations etc.
    k_max: Optional[float] = eqx.field(default=None, static=True)

    # Angles / thresholds (per shim). Fixed hyperparams here.
    thetas_ss: jax.Array = eqx.field(static=True)  # (H, S)
    thresh: jax.Array = eqx.field(static=True)     # (H, S)
    
    # Stretch stiffness (scalar or (H,S)); fixed hyperparam
    k_stretch: jax.Array = eqx.field(static=True)

    # Normalizations (fixed)
    norm_pos: float = eqx.field(init=False, static=True)
    norm_angle: float = eqx.field(init=False, static=True)
    norm_force: float = eqx.field(init=False, static=True)
    norm_torque: float = eqx.field(init=False, static=True)

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
            self.k_soft = jnp.ones((H, S), jnp.float32) if k_soft is None else jnp.asarray(k_soft,  jnp.float32)
            self.k_stiff = jnp.ones((H, S), jnp.float32) if k_stiff is None else jnp.asarray(k_stiff, jnp.float32)
            self.k = None
            self.k_max = float(jnp.max(self.k_stiff))
            self.k_stretch = jnp.asarray(stretch_scale * self.k_max, jnp.float32)

        elif k_type == "Experimental":
            self.k_soft = None
            self.k_stiff = None
            thetas, torques, ks, torque_of_theta, k_of_theta = file_funcs.build_torque_stiffness_from_file(
                file_name, savgol_window=9
            )
            self.k_max = float(jnp.max(ks))
            self.k = k_of_theta                           # callable -> static
            self.k_stretch = jnp.asarray(stretch_scale * jnp.max(ks), jnp.float32)

        else:
            raise ValueError(f"Unknown k_type: {k_type}")

        self.thetas_ss = (jnp.ones((H, S), jnp.float32) if thetas_ss is None
                          else jnp.asarray(thetas_ss, jnp.float32))
        self.thresh = (jnp.ones((H, S), jnp.float32) if thresh is None
                       else jnp.asarray(thresh, jnp.float32))

        # normalizations for update values
        self.norm_pos = float(Strctr.hinges*Strctr.L)
        self.norm_angle = float(np.pi/2)
        self.norm_force = float(self.k_max*(np.pi/2))
        self.norm_torque = float(self.k_max*(np.pi/2)*Strctr.L)
