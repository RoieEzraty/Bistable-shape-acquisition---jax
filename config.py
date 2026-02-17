from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


# -----------------------------
# Structure and initial params
# -----------------------------

# MATERIAL = "numerical"
# MATERIAL = "Leon_plastic"
# MATERIAL = "Leon_metal"
MATERIAL = "Roie_metal"


# -----------------------------
# Structure and initial params
# -----------------------------
@dataclass(frozen=True)
class StructureConfig:
    # H: int = 5  # Hinges
    H: int = 4  # Hinges
    S: int = 1  # Shims per hinge
    # Nin: int = 3  # tip position in (x, y) and its angle
    # Nout: int = 3  # Fx, Fy, torque, all on tip
    # Nin: int = 3  # tip position in (x, y) and its angle at left side
    # Nout: int = 3  # x, y, theta of tip
    # Nin: int = 3  # x, y, theta of tip
    # Nout: int = 2  # Fx, Fy
    Nin: int = 2  # total and tip angles
    Nout: int = 2  # Fx Fy transformed into total and tip angle forces


# -----------------------------
# Material / variables
# -----------------------------
@dataclass(frozen=True)
class VariablesConfig:
    material: str = MATERIAL  # "Leon_plastic" | "Leon_metal" | "numerical" | "Roie_metal"

    # chosen per material
    k_type: str = field(init=False)
    tau_file: str | None = field(init=False)
    thetas_ss: float = field(init=False)
    thresh: float = field(init=False)
    k_soft: str | None = field(init=False)
    k_stiff: str | None = field(init=False)

    def __post_init__(self):
        if self.material == "Leon_plastic":
            object.__setattr__(self, "k_type", "Leon_plastic_txt")
            object.__setattr__(self, "tau_file", "Roee_offset3mm_dl75.txt")
            object.__setattr__(self, "thetas_ss", 1.03312)  # not used in experimental
            object.__setattr__(self, "thresh", 1.96257)
            object.__setattr__(self, "k_soft", None)
            object.__setattr__(self, "k_stiff", None)
        elif self.material == "Leon_metal":
            object.__setattr__(self, "k_type", "Leon_metal_txt")
            object.__setattr__(self, "tau_file", "Roee_metal_offset3mm_dl75.txt")
            object.__setattr__(self, "thetas_ss", 1.227)  # not used in experimental
            object.__setattr__(self, "thresh", 1.693)
            object.__setattr__(self, "k_soft", None)
            object.__setattr__(self, "k_stiff", None)
        elif self.material == "Roie_metal":
            object.__setattr__(self, "k_type", "Roie_metal_csv")
            object.__setattr__(self, "tau_file", "Roie_metal_singleMylar_short.csv")
            object.__setattr__(self, "thetas_ss", 0.91)  # not used in experimental
            object.__setattr__(self, "thresh", 1.58)
            object.__setattr__(self, "k_soft", None)
            object.__setattr__(self, "k_stiff", None)
        elif self.material == "numerical":
            object.__setattr__(self, "k_type", "Numerical")
            object.__setattr__(self, "tau_file", None)
            object.__setattr__(self, "thetas_ss", 1/2)
            object.__setattr__(self, "thresh", 1)
            object.__setattr__(self, "k_soft", 1.0)
            object.__setattr__(self, "k_stiff", 1.5)
        else:
            raise ValueError(f"Unknown material: {self.material}")

    # ADMET stress-strain tests from 2025Oct by Roie
    exp_start: float = 280*1e-3  # tip position start, not accounting for 2 first edges [m]
    exp_start = exp_start*0.99  # make sure to not stretch too much in simulation
    distance: float = 140*1e-3  # how much the arms compressed, [m]

    # numerical stability
    contact_scale: float = 100  # max experimental torque and torque upon edge contact ratio, for numerical stability


# -----------------------------
# Equilibrium solver
# -----------------------------
@dataclass(frozen=True)
class EquilibriumConfig:
    material: str = MATERIAL

    # chosen per material
    k_stretch_ratio: float = field(init=False)
    T_eq: float = field(init=False)
    damping: float = field(init=False)
    mass: float = field(init=False)

    # independent knobs
    calc_through_energy: bool = False
    rand_key_Eq: int = 3
    pos_noise: float = 0.1
    vel_noise: float = 1.0
    ramp_pos: bool = True
    r_intersect_factor: float = 0.1  # best one 2026Feb8
    k_intersect_factor: float = 10000.0  # best one 2026Feb8
    # tolerance: float = 1e-8
    tolerance: float = 1e-4

    def __post_init__(self):
        if self.material in {"Leon_plastic", "numerical"}:
            object.__setattr__(self, "k_stretch_ratio", 2e4)
            object.__setattr__(self, "T_eq", 0.04)
            object.__setattr__(self, "damping", 4.0)
            object.__setattr__(self, "mass", 5e-3)
        elif self.material in {"Leon_metal", "Roie_metal"}:
            object.__setattr__(self, "k_stretch_ratio", 2e4)
            object.__setattr__(self, "T_eq", 0.04)
            object.__setattr__(self, "damping", 4.0)
            object.__setattr__(self, "mass", 12e-3)
        else:
            raise ValueError(f"Unknown material: {self.material}")


# -----------------------------
# Training / supervisor
# -----------------------------
@dataclass(frozen=True)
class TrainingConfig:
    T: int = 42  # total training set time (not time to reach equilibrium during every step)

    # desired_buckle_type: str = 'random'
    # desired_buckle_type: str = 'opposite'
    # desired_buckle_type: str = 'straight'
    desired_buckle_type: str = 'specified'
    
    if desired_buckle_type == 'random':
        desired_buckle_rand_key: int = 169  # key for seed of random sampling of buckle pattern
    elif desired_buckle_type == 'specified':
        # desired_buckle_pattern: tuple = (1, -1, -1, -1, -1)  # which shims should be buckled up, initially
        desired_buckle_pattern: tuple = (1, -1, -1, -1)  # which shims should be buckled up, initially
        # desired_buckle_pattern: tuple = (-1, 1, 1, 1)  # which shims should be buckled up, initially

    # init_buckle_pattern: tuple = (-1, -1, -1, -1, 1)  # which shims should be buckled up, initially
    init_buckle_pattern: tuple = (-1, -1, -1, -1)  # which shims should be buckled up, initially
    # init_buckle_pattern: tuple = (1, 1, 1, -1)  # which shims should be buckled up, initially

    # dataset_sampling: str = 'uniform'  # random uniform vals for x, y, angle
    dataset_sampling: str = 'specified'  # constant
    # dataset_sampling: str = 'tile'  # constant
    # dataset_sampling = 'almost flat'  # flat piece, single measurement
    # dataset_sampling = 'stress strain'

    # # tip values to buckle shims - 'BEASTAL' for the BEASTAL scheme, else 'one_to_one'
    update_scheme: str = 'one_to_one'  # direct normalized loss, equal to num of outputs
    # update_scheme: str = 'radial_one_to_one'  # evolve tip angle and large radius due to instantaneous loss
    # update_scheme: str = 'BEASTAL'  # update using the BEASTAL scheme (with pseudoinverse of the incidence matrix).
    # update_scheme: str = 'BEASTAL_no_pinv'  # update using (y_j)(Loss_j), no psuedo inv of the incidence matrix.
    # update_scheme: str = 'radial_halfway_BEASTAL'  # evolve tip angle and large radius due to instantaneous loss
    # update_scheme: str = 'radial_BEASTAL'  # update using BEASTAL (pseudoinverse of 2x2 incidence matrix),
                                             # calculated in total and tip angles

    normalize_step: bool = True
    # normalize_step: bool = False

    if update_scheme == 'radial_BEASTAL' and not normalize_step:
        alpha: float = 1.0  # learning rate
    elif normalize_step:
        alpha: float = 0.1
    else:
        alpha: float = 0.12  # learning rate

    loss_type: str = 'cartesian'
    # loss_type: str = 'Fx_and_tip_torque'

    control_tip_pos: bool = True  # imposed tip position in measurement and update. If False, tip is free
    control_tip_angle: bool = True  # impose tip angle in measurement and update. If False, imposed tip pos but free to ratoate
    control_first_edge: bool = True  # if True, fix nodes (0, 1) to zero. if Flase, just the first

    rand_key_dataset: int = 7  # for random sampling of dataset, if dataset_sampling is True

    convert_pos = 1000  # convert [m] to [mm]
    convert_angle = 180/np.pi  # convert rad to deg
    convert_F = 1  # already in [mN]


# -----------------------------
# A single top-level config
# -----------------------------
@dataclass(frozen=True)
class ExperimentConfig:
    Strctr: StructureConfig = StructureConfig()
    Variabs: VariablesConfig = VariablesConfig()
    Eq: EquilibriumConfig = EquilibriumConfig()
    Train: TrainingConfig = TrainingConfig()


# Default instance you can import directly
CFG = ExperimentConfig()
