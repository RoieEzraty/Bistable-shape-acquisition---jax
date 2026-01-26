from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from typing import Tuple, List
from numpy import array, zeros
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from StructureClass import StructureClass
    from config import ExperimentConfig

import file_funcs

# ===================================================
# Class - User Variables: stiffnesses, lengths, etc.
# ===================================================


class VariablesClass(eqx.Module):
    """
    Material / stiffness parameters and normalization scales for the bistable chain.

    The class supports two operation modes:

    - ``k_type == "Numerical"``:
        * Piecewise-linear Hookean torque model.
        * Uniform soft / stiff rotational spring constants per hinge and shim.
    - ``k_type == "Experimental"``:
        * Torque and stiffness are obtained from Leon Kamp's experimental τ(θ) curve
          using `file_funcs.build_torque_and_k_from_file`.

    Attributes
    ----------
    k_type          - Which torque/stiffness model to use. Taken from ``CFG.Variabs.k_type``.
    k_soft, k_stiff - ndarray or None, shape (H, S). Soft and stiff rotational spring constants per hinge and shim
                      (used only in the "Numerical" mode).
    torque          - callable or None. Function: hinge angle θ → torque τ(θ). Only set in the "Experimental" mode.
    thetas_ss       - ndarray, shape (H, S). Switching angle(s) θ_ss for each hinge and shim. For "Numerical"
    thresh          - ndarray, shape (H, S). Threshold angles to buckle shims, broadcast from config. for "Numerical"
    k_stretch       - array-like (scalar or (E,)). Stretch stiffness for edges, used by the equilibrium solver.
                      It is a large multiple of ``k_max`` so that edges are nearly inextensible.
    norm_pos        - float. Position normalization scale (used by the supervisor), set to ``Strctr.L``.
    norm_angle      - float. Angle normalization scale, set to π/2.
    norm_torque     - float. Torque normalization scale, k_max * norm_angle for numerical or mean torque for experimental
    norm_force      - float. Force normalization scale, norm_torque / norm_pos.   
    """

    # --- configuration / mode ---
    k_type: str = eqx.field(static=True)  # "Numerical" | "Experimental"

    # if using numerical values for springs
    k_soft: Optional[NDArray[np.float_]] = eqx.field(default=None, static=True)   # (H, S)
    k_stiff: Optional[NDArray[np.float_]] = eqx.field(default=None, static=True)  # (H, S)

    # torque is callables when using experimental data
    torque: Optional[Callable[[NDArray[np.float_]], NDArray[np.float_]]] = eqx.field(default=None, static=True)

    # Angles / thresholds (per shim). Fixed hyperparams here.
    thetas_ss: Optional[NDArray[np.float_]] = eqx.field(default=None, static=True)  # (H, S)
    thresh: Optional[NDArray[np.float_]] = eqx.field(default=None, static=True)     # (H, S)  
    
    # Stretch stiffness (scalar or (H,S)); fixed hyperparam
    k_stretch: NDArray[np.float_] = eqx.field(static=True)

    # Normalizations (fixed)
    norm_pos: float = eqx.field(init=False, static=True)
    norm_angle: float = eqx.field(init=False, static=True)
    norm_force: float = eqx.field(init=False, static=True)
    norm_torque: float = eqx.field(init=False, static=True)

    def __init__(self, Strctr: "StructureClass", CFG):
        """
        Parameters
        ----------
        Strctr : StructureClass.
        CFG : ExperimentConfig.
        """
        H, S, L = Strctr.hinges, Strctr.shims, Strctr.L
        
        # normalizations for update values
        self.norm_pos = float(L)
        self.norm_angle = float(np.pi/2)

        self.k_type = CFG.Variabs.k_type  # static

        if self.k_type == "Numerical":  # numerical model - Hookean torque
            self.k_soft = CFG.Variabs.k_soft * np.array((H, S), dtype=jnp.float32)
            self.k_stiff = CFG.Variabs.k_stiff * np.array((H, S), dtype=jnp.float32)
            k_max = float(np.max(self.k_stiff))  # Maximum stiffness over all hinges/shims (or from experimental k-grid).
                                                 # Used for normalization and for computing stretch stiffness.
            self.k_stretch = np.asarray(CFG.Eq.k_stretch_ratio * k_max, np.float32)
            thetas_ss_scalar = CFG.Variabs.thetas_ss  # if Experimental, not used
            # Broadcast scalar thresholds to full (H, S) arrays
            self.thetas_ss = thetas_ss_scalar * np.ones((H, S), np.float32)
            thresh_scalar = CFG.Variabs.thresh
            self.norm_torque = float(self.k_max*self.norm_angle)
        elif self.k_type in {"Experimental_plastic", "Experimental_metal"}:  # Leon's shim
            self.k_soft = None
            self.k_stiff = None
            # Load τ(θ) and k(θ) from experimental file
            _, _, ks, tau_of_theta, _ = file_funcs.build_torque_and_k_from_file(CFG.Variabs.tau_file, savgol_window=9,
                                                                                contact=True,
                                                                                contact_scale=CFG.Variabs.contact_scale)
            self.torque = tau_of_theta
            self.k_stretch = np.asarray(CFG.Eq.k_stretch_ratio * np.max(ks), np.float32)
            thresh_scalar = CFG.Variabs.thresh
            tau_plus = float(np.abs(self.torque(self.norm_angle)))
            tau_minus = float(np.abs(self.torque(-self.norm_angle)))
            self.norm_torque = np.mean([tau_plus, tau_minus])
        else:
            raise ValueError(f"Unknown k_type: {CFG.Variabs.k_type}")
        # Broadcast scalar thresholds to full (H, S) arrays
        self.thresh = thresh_scalar * np.ones((H, S), np.float32)
        self.norm_force = self.norm_torque / self.norm_pos
