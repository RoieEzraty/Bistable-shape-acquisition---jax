from __future__ import annotations
import equinox as eqx
import jax
import jax.numpy as jnp

from typing import Tuple, List
from numpy import array, zeros
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional

if TYPE_CHECKING:
    from StructureClass import StructureClass


# ===================================================
# Class - User Variables: stiffnesses, lengths, etc.
# ===================================================


class VariablesClass(eqx.Module):
    """Class with Bistable buckle shape variables.

    Attributes given by the user:
        `k_soft`: jax.Array: torque constants for each hinge in soft direction
        `k_stiff`: jax.Array: torque constants for each hinge in soft direction

    Attributes computed by the class:
        
    """
    k_soft: jax.Array = eqx.field(init=False)
    """`(hinges, shims)` stiffnesses in soft direction"""

    k_stiff: jax.Array = eqx.field(init=False)
    """`(hinges, shims)` stiffnesses in stiff direction"""
    
    thetas_ss: jax.Array = eqx.field(init=False)
    """`(hinges, shims)` rest angles of hinges"""

    thresh: jax.Array = eqx.field(init=False)
    """`(hinges, shims)` threshold angle to buckle shim"""
    
    k_stretch: jax.Array = eqx.field(init=False)
    """`(1, )` stiffnesses of rods, very large so rods are stiff""" 

    def __init__(self,
                 Strctr: "StructureClass",
                 k_soft: jax.Array = None,
                 k_stiff: jax.Array = None,
                 thetas_ss: jax.Array = None,
                 thresh: jax.Array = None,
                 stretch_scale: float = 1e3) -> None:
    
        H, S = Strctr.hinges, Strctr.shims
        
        self.k_soft = jnp.ones((H, S), jnp.float32) if k_soft is None else jnp.asarray(k_soft,  jnp.float32)
        self.k_stiff = jnp.ones((H, S), jnp.float32) if k_stiff is None else jnp.asarray(k_stiff, jnp.float32)
        self.thetas_ss = jnp.ones((H, S), jnp.float32) if thetas_ss is None else jnp.asarray(thetas_ss, jnp.float32)
        self.thresh = jnp.ones((H, S), jnp.float32) if thresh is None else jnp.asarray(thresh, jnp.float32)

        # A single stretch stiffness (applied to every edge)
        self.k_stretch = jnp.asarray(stretch_scale * jnp.max(k_stiff), jnp.float32)
