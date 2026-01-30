from __future__ import annotations
from dataclasses import dataclass, field


# -----------------------------
# Structure and initial params
# -----------------------------

# MATERIAL = "numerical"
# MATERIAL = "plastic"
MATERIAL = "metal"


# -----------------------------
# Structure and initial params
# -----------------------------
@dataclass(frozen=True)
class StructureConfig:
    # H: int = 5  # Hinges
    H: int = 4  # Hinges
    S: int = 1  # Shims per hinge
    Nin: int = 3  # tip position in (x, y) and its angle
    Nout: int = 3  # Fx, Fy, torque, all on tip


# -----------------------------
# Material / variables
# -----------------------------
# @dataclass(frozen=True)
# class VariablesConfig:
#     k_type = 'Experimental_metal'  # Leon's shim
#     # k_type = 'Numerical'  # numerical model - Hookean torque

#     if k_type == 'Numercial':  # For numerical torque model, not experimental Leon stuff
#         k_soft_uniform = 1.0
#         k_stiff_uniform = 1.5
#         thetas_ss_uniform = 1/2
#         thresh_uniform = 1
#     elif k_type == 'Experimental_plastic':  # For experimental torque model
#         tau_file: str | None = "Roee_offset3mm_dl75.txt"  # relative path
#         thetas_ss_exp: float = 1.03312
#         thresh_exp: float = 1.96257
#     elif k_type == 'Experimental_metal':  # For experimental torque model
#         tau_file: str | None = "Roee_metal_offset3mm_dl75.txt"  # relative path
#         thetas_ss_exp: float = 1.227
#         thresh_exp: float = 1.693

#     # ADMET stress-strain tests from 2025Oct by Roie
#     exp_start: float = 280*1e-3  # tip position start, not accounting for 2 first edges [m]
#     exp_start = exp_start*0.99  # make sure to not stretch too much in simulation
#     distance: float = 140*1e-3  # how much the arms compressed, [m]

#     # numerical stability
#     contact_scale: float = 100  # max experimental torque and torque upon edge contact ratio, for numerical stability


@dataclass(frozen=True)
class VariablesConfig:
    material: str = MATERIAL  # "plastic" | "metal" | "numerical"

    # common
    contact_scale: float = 100

    # chosen per material
    k_type: str = field(init=False)
    tau_file: str | None = field(init=False)
    thetas_ss: float = field(init=False)
    thresh: float = field(init=False)
    k_soft: str | None = field(init=False)
    k_stiff: str | None = field(init=False)

    def __post_init__(self):
        if self.material == "plastic":
            object.__setattr__(self, "k_type", "Experimental_plastic")
            object.__setattr__(self, "tau_file", "Roee_offset3mm_dl75.txt")
            object.__setattr__(self, "thetas_ss", 1.03312)
            object.__setattr__(self, "thresh", 1.96257)
            object.__setattr__(self, "k_soft", None)
            object.__setattr__(self, "k_stiff", None)
        elif self.material == "metal":
            object.__setattr__(self, "k_type", "Experimental_metal")
            object.__setattr__(self, "tau_file", "Roee_metal_offset3mm_dl75.txt")
            object.__setattr__(self, "thetas_ss", 1.227)
            object.__setattr__(self, "thresh", 1.693)
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
# @dataclass(frozen=True)
# class EquilibriumConfig:
#     if k_type in {"Experimental_plastic", "numerical"}:
#         k_stretch_ratio: float = 2e4  # Stretch force to Torque force ratio, to make edges stiff but not inifinitely stiff.
#         T_eq: float = 0.04  # total time for equilibrium calculation, [s]
#         damping = 4.0  # damping coefficient for right-hand-side of ODE. Should be something*sqrt(k*m)
#         mass: float = 5e-3  # divides right-hand-side of ODE, [kg]
#         tolerance: float = 1e-8  # for ODE
#     elif k_type == "Experimental_metal":
#         k_stretch_ratio: float = 2e4  # Stretch force to Torque force ratio, to make edges stiff but not inifinitely stiff.
#         T_eq: float = 0.04  # total time for equilibrium calculation, [s]
#         damping = 4.0  # damping coefficient for right-hand-side of ODE. Should be something*sqrt(k*m)
#         mass: float = 5e-3  # divides right-hand-side of ODE, [kg]
#         tolerance: float = 1e-8  # for ODE
#     calc_through_energy: bool = False  # If False, calculate through torque and stretch forces
#     rand_key_Eq = 2  # random key for noise on initial positions and velocities
#     pos_noise = 0.1  # noise on initial positions
#     vel_noise = 1.0  # noise on initial velocities
#     ramp_pos = True  # ramp up tip position from previous to next, during equilibrium calculation

@dataclass(frozen=True)
class EquilibriumConfig:
    material: str = MATERIAL

    # chosen per material
    k_stretch_ratio: float = field(init=False)
    T_eq: float = field(init=False)
    damping: float = field(init=False)
    mass: float = field(init=False)
    tolerance: float = field(init=False)

    # independent knobs
    calc_through_energy: bool = False
    rand_key_Eq: int = 3
    pos_noise: float = 0.1
    vel_noise: float = 1.0
    ramp_pos: bool = True

    # scale from mN to N
    scale_to_N: float = 1.00

    def __post_init__(self):
        if self.material in {"plastic", "numerical"}:
            object.__setattr__(self, "k_stretch_ratio", 2e4)
            object.__setattr__(self, "T_eq", 0.04)
            object.__setattr__(self, "damping", 4.0)
            object.__setattr__(self, "mass", 5e-3)
            object.__setattr__(self, "tolerance", 1e-8)
        elif self.material == "metal":
            object.__setattr__(self, "k_stretch_ratio", 2e4)
            object.__setattr__(self, "T_eq", 0.04)
            object.__setattr__(self, "damping", 4.0)
            object.__setattr__(self, "mass", 12e-3)
            object.__setattr__(self, "tolerance", 1e-8)
        else:
            raise ValueError(f"Unknown material: {self.material}")


# -----------------------------
# Training / supervisor
# -----------------------------
@dataclass(frozen=True)
class TrainingConfig:
    T: int = 14  # total training set time (not time to reach equilibrium during every step)
    alpha: float = 0.1  # learning rate

    # desired_buckle_type: str = 'random'
    # desired_buckle_type: str = 'opposite'
    # desired_buckle_type: str = 'straight'
    desired_buckle_type: str = 'specified'
    
    if desired_buckle_type == 'random':
        desired_buckle_rand_key: int = 169  # key for seed of random sampling of buckle pattern
    elif desired_buckle_type == 'specified':
        # desired_buckle_pattern: tuple = (1, -1, -1, -1, -1)  # which shims should be buckled up, initially
        desired_buckle_pattern: tuple = (1, -1, 1, -1)  # which shims should be buckled up, initially

    # dataset_sampling: str = 'uniform'  # random uniform vals for x, y, angle
    dataset_sampling: str = 'specified'  # constant
    # dataset_sampling = 'almost flat'  # flat piece, single measurement
    # dataset_sampling = 'stress strain'

    # # tip values to buckle shims - 'BEASTAL' for the BEASTAL scheme, else 'one_to_one'
    update_scheme: str = 'one_to_one'  # direct normalized loss, equal to num of outputs
    # update_scheme: str = 'BEASTAL'  # update using the BEASTAL scheme (with pseudoinverse of the incidence matrix).
    # update_scheme: str = 'BEASTAL_no_pinv'  # update using (y_j)(Loss_j), no psuedo inv of the incidence matrix.

    loss_type: str = 'cartesian'
    # loss_type: str = 'Fx_and_tip_torque'

    control_tip_pos: bool = True  # imposed tip position in measurement and update. If False, tip is free
    control_tip_angle: bool = True  # impose tip angle in measurement and update. If False, imposed tip pos but free to ratoate
    control_first_edge: bool = True  # if True, fix nodes (0, 1) to zero. if Flase, just the first
    # init_buckle_pattern: tuple = (-1, -1, -1, -1, 1)  # which shims should be buckled up, initially
    init_buckle_pattern: tuple = (-1, 1, -1, 1)  # which shims should be buckled up, initially

    rand_key_dataset: int = 7  # for random sampling of dataset, if dataset_sampling is True


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
