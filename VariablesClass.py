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

    def __init__(self,
                 Strctr: "StructureClass",
                 k_type: str = "Numerical",
                 k_soft: NDArray[np.float_] | None = None,
                 k_stiff: NDArray[np.float_] | None = None,
                 thetas_ss: NDArray[np.float_] | None = None,
                 thresh: NDArray[np.float_] | None = None,
                 stretch_scale: float = 1e3,
                 file_name: str | None = None):
        H, S = Strctr.hinges, Strctr.shims
        self.k_type = k_type  # static

        if k_type == "Numerical":
            self.k_soft = np.ones((H, S), np.float32) if k_soft is None else np.asarray(k_soft,  np.float32)
            self.k_stiff = np.ones((H, S), np.float32) if k_stiff is None else np.asarray(k_stiff, np.float32)
            self.k = None
            self.k_max = float(np.max(self.k_stiff))
            self.k_stretch = np.asarray(stretch_scale * self.k_max, np.float32)

        elif k_type == "Experimental":
            self.k_soft = None
            self.k_stiff = None
            thetas, torques, ks, torque_of_theta, k_of_theta = file_funcs.build_torque_stiffness_from_file(
                file_name, savgol_window=9
            )
            self.k_max = float(np.max(ks))
            self.k = k_of_theta 
            self.torque = torque_of_theta
            self.k_stretch = np.asarray(stretch_scale * np.max(ks), np.float32)

        else:
            raise ValueError(f"Unknown k_type: {k_type}")

        self.thetas_ss = (np.ones((H, S), np.float32) if thetas_ss is None
                          else np.asarray(thetas_ss, np.float32))
        self.thresh = (np.ones((H, S), np.float32) if thresh is None
                       else np.asarray(thresh, np.float32))

        # normalizations for update values
        self.norm_pos = float(Strctr.hinges*Strctr.L)
        self.norm_angle = float(np.pi/2)
        self.norm_force = float(self.k_max*(np.pi/2))
        self.norm_torque = float(self.k_max*(np.pi/2)*Strctr.L)
