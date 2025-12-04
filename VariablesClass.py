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
    k_soft: Optional[NDArray[np.float_]] = eqx.field(default=None, static=True)   # (H, S)
    k_stiff: Optional[NDArray[np.float_]] = eqx.field(default=None, static=True)  # (H, S)

    # k and torque are callables when using experimental data
    k: Optional[Callable[[NDArray[np.float_]], NDArray[np.float_]]] = eqx.field(default=None, static=True)
    torque: Optional[Callable[[NDArray[np.float_]], NDArray[np.float_]]] = eqx.field(default=None, static=True)

    # maximal spring constant for normalizations etc.
    k_max: Optional[float] = eqx.field(default=None, static=True)

    # Angles / thresholds (per shim). Fixed hyperparams here.
    thetas_ss: NDArray[np.float_] = eqx.field(static=True)  # (H, S)
    thresh: NDArray[np.float_] = eqx.field(static=True)     # (H, S)
    
    # Stretch stiffness (scalar or (H,S)); fixed hyperparam
    k_stretch: NDArray[np.float_] = eqx.field(static=True)

    # Normalizations (fixed)
    norm_pos: float = eqx.field(init=False, static=True)
    norm_angle: float = eqx.field(init=False, static=True)
    norm_force: float = eqx.field(init=False, static=True)
    norm_torque: float = eqx.field(init=False, static=True)

    def __init__(self, Strctr: "StructureClass", CFG):
        H, S, L = Strctr.hinges, Strctr.shims, Strctr.L
        # normalizations for update values
        self.norm_pos = float(H*L)
        self.norm_angle = float(np.pi/2)

        self.k_type = CFG.Variabs.k_type  # static

        if CFG.Variabs.k_type == "Numerical":
            self.k_soft = CFG.Variabs.k_soft_uniform * np.array((H, S), dtype=jnp.float32)
            self.k_stiff = CFG.Variabs.k_stiff_uniform * np.array((H, S), dtype=jnp.float32)
            self.k = None
            self.k_max = float(np.max(self.k_stiff))
            self.k_stretch = np.asarray(CFG.Eq.k_stretch_ratio * self.k_max, np.float32)
            thetas_ss = CFG.Variabs.thetas_ss_uniform
            thresh = CFG.Variabs.thresh_uniform
            self.norm_torque = float(self.k_max*self.norm_angle)
            self.norm_force = float(self.k_max*self.norm_angle/self.norm_pos)

        elif CFG.Variabs.k_type == "Experimental":
            self.k_soft = None
            self.k_stiff = None
            thetas, torques, ks, torque_of_theta, k_of_theta = file_funcs.build_torque_and_k_from_file(CFG.Variabs.tau_file,
                                                                                                       savgol_window=9,
                                                                                                       contact=True,
                                                                                                       contact_scale=2)
            self.k_max = float(np.max(ks))
            self.k = k_of_theta 
            self.torque = torque_of_theta
            self.k_stretch = np.asarray(CFG.Eq.k_stretch_ratio * np.max(ks), np.float32)
            thetas_ss = CFG.Variabs.thetas_ss_exp
            thresh = CFG.Variabs.thresh_exp

            self.norm_torque = np.mean([np.abs(self.torque(self.norm_angle)), np.abs(self.torque(-self.norm_angle))])
            self.norm_force = self.norm_torque / self.norm_pos

        else:
            raise ValueError(f"Unknown k_type: {CFG.Variabs.k_type}")

        self.thetas_ss = thetas_ss * np.ones((H, S), np.float32)
        self.thresh = thresh * np.ones((H, S), np.float32)
    
        
