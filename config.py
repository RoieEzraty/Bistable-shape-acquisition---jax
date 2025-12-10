from __future__ import annotations
from dataclasses import dataclass


# -----------------------------
# Structure and initial params
# -----------------------------
@dataclass(frozen=True)
class StructureConfig:
    H: int = 5  # Hinges
    S: int = 1  # Shims per hinge
    Nin: int = 3  # tip position in (x, y) and its angle
    Nout: int = 3  # Fx, Fy, torque, all on tip


# -----------------------------
# Material / variables
# -----------------------------
@dataclass(frozen=True)
class VariablesConfig:
    k_type = 'Experimental'  # Leon's shim
    # k_type = 'Numerical'  # numerical model - Hookean torque

    if k_type == 'Numercial':  # For numerical torque model, not experimental Leon stuff
        k_soft_uniform = 1.0
        k_stiff_uniform = 1.5
        thetas_ss_uniform = 1/2
        thresh_uniform = 1
    elif k_type == 'Experimental':  # For experimental torque model
        tau_file: str | None = "Roee_offset3mm_dl75.txt"  # relative path
        thetas_ss_exp: float = 1.03312
        thresh_exp: float = 1.96257

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
    k_stretch_ratio: float = 2e4  # Stretch force to Torque force ratio, to make edges stiff but not inifinitely stiff.
    T_eq: float = 0.04  # total time for equilibrium calculation, [s]
    damping = 4.0  # damping coefficient for right-hand-side of ODE. Should be something*sqrt(k*m)
    mass: float = 5e-3  # divides right-hand-side of ODE, [kg]
    tolerance: float = 1e-8  # for ODE
    calc_through_energy: bool = False  # If False, calculate through torque and stretch forces
    rand_key_Eq = 2  # random key for noise on initial positions and velocities
    pos_noise = 0.1  # noise on initial positions
    vel_noise = 1.0  # noise on initial velocities
    ramp_pos = True  # ramp up tip position from previous to next, during equilibrium calculation


# -----------------------------
# Training / supervisor
# -----------------------------
@dataclass(frozen=True)
class TrainingConfig:
    T: int = 64  # total training set time (not time to reach equilibrium during every step)
    alpha: float = 0.2  # learning rate

    # desired_buckle_type: str = 'random'
    # desired_buckle_type: str = 'opposite'
    # desired_buckle_type: str = 'straight'
    desired_buckle_type: str = 'specified'
    
    if desired_buckle_type == 'random':
        desired_buckle_rand_key: int = 169  # key for seed of random sampling of buckle pattern
    elif desired_buckle_type == 'specified':
        desired_buckle_pattern: tuple = (1, -1, -1, -1, -1)  # which shims should be buckled up, initially

    dataset_sampling: str = 'uniform'  # random uniform vals for x, y, angle
    # dataset_sampling: str = 'specified'  # random uniform vals for x, y, angle
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
    init_buckle_pattern: tuple = (-1, -1, -1, -1, 1)  # which shims should be buckled up, initially

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
