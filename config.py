from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


# -----------------------------
# Relating to all
# -----------------------------

# MATERIAL = "numerical"
# MATERIAL = "Leon_plastic"
# MATERIAL = "Leon_metal"
MATERIAL = "Roie_metal"

# HINGES: int = 4  # Hinges
# HINGES: int = 8  # Hinges
HINGES: int = 5  # Hinges


# -----------------------------
# Structure and initial params
# -----------------------------
@dataclass(frozen=True)
class StructureConfig:
    H: int = HINGES
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
            object.__setattr__(self, "tau_file", "single_hinge_files/Roee_offset3mm_dl75.txt")
            object.__setattr__(self, "thetas_ss", 1.03312)  # not used in experimental
            object.__setattr__(self, "thresh", 1.96257)
            object.__setattr__(self, "k_soft", None)
            object.__setattr__(self, "k_stiff", None)
        elif self.material == "Leon_metal":
            object.__setattr__(self, "k_type", "Leon_metal_txt")
            object.__setattr__(self, "tau_file", "single_hinge_files/Roee_metal_offset3mm_dl75.txt")
            object.__setattr__(self, "thetas_ss", 1.227)  # not used in experimental
            object.__setattr__(self, "thresh", 1.693)
            object.__setattr__(self, "k_soft", None)
            object.__setattr__(self, "k_stiff", None)
        elif self.material == "Roie_metal":
            object.__setattr__(self, "k_type", "Roie_metal_csv")
            # object.__setattr__(self, "tau_file", "Roie_metal_singleMylar_short.csv")
            # object.__setattr__(self, "tau_file", "single_hinge_files/Stress_Strain_steel_1myl1tp_short.csv")
            # object.__setattr__(self, "thresh", 1.53)  # Feb23 realistically from just before red south
            # object.__setattr__(self, "tau_file", "single_hinge_files/Stress_Strain_1myl1tp_otherEnd_short.csv")
            # object.__setattr__(self, "tau_file", "single_hinge_files/Mar9_filled_average.csv")
            # object.__setattr__(self, "tau_file", "single_hinge_files/Mar12_dl90.csv")  # up to May22
            # object.__setattr__(self, "thresh", 1.24)  # Mar12 dl90
            # object.__setattr__(self, "tau_file", "single_hinge_files/May22_old_dl90_toughend.csv")
            # object.__setattr__(self, "thresh", 1.1)  # May22_old_dl90_toughend
            object.__setattr__(self, "tau_file", "single_hinge_files/May24_dl90_2ndEnd.csv")  # May24 2nd end (notated on chain itself)
            object.__setattr__(self, "thresh", 1.15)  # May24 2nd end (notated on chain itself)
            # object.__setattr__(self, "tau_file", "single_hinge_files/May24_dl90_1stEnd.csv")   # May24 1st end (notated on chain itself)
            # object.__setattr__(self, "thresh", 0.96)  # May24 1st end (notated on chain itself)
            # object.__setattr__(self, "thresh", 1.99)  # Feb23
            # object.__setattr__(self, "thresh", 1.58)
            # object.__setattr__(self, "thresh", 1.9)  # Feb22 measurements from just before Red South
            object.__setattr__(self, "thetas_ss", 0.91)  # not used in experimental
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
    # contact_scale: float = 1  # max experimental torque and torque upon edge contact ratio, for numerical stability


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
    # pos_noise: float = 0.1  # best one 2026Feb8
    pos_noise: float = 0.002
    # vel_noise: float = 1.0  # best one 2026Feb8
    vel_noise: float = 0.1
    ramp_pos: bool = True
    r_intersect_factor: float = 0.1  # best one 2026Feb8
    # r_intersect_factor: float = 0.25  # 2026Apr16
    k_intersect_factor: float = 10000.0  # best one 2026Feb8
    # k_intersect_factor: float = 100000.0
    tolerance: float = 1e-4  # best one 2026Feb8
    # tolerance: float = 1e-6
    maxsteps: int = 16000  # 2026Apr21

    def __post_init__(self):
        if self.material in {"Leon_plastic", "numerical"}:
            object.__setattr__(self, "k_stretch_ratio", 2e4)
            object.__setattr__(self, "T_eq", 0.04)
            object.__setattr__(self, "damping", 4.0)
            object.__setattr__(self, "mass", 5e-3)
        elif self.material in {"Leon_metal", "Roie_metal"}:
            # object.__setattr__(self, "k_stretch_ratio", 9e3)  # best one 2026Feb8
            object.__setattr__(self, "k_stretch_ratio", 1e5)  # best one 2026Feb8
            # object.__setattr__(self, "T_eq", 0.04)  # best one 2026Feb8
            object.__setattr__(self, "T_eq", 0.64)  # 2026Apr21 good?
            object.__setattr__(self, "damping", 8.0)  # best one 2026Feb8
            object.__setattr__(self, "mass", 12e-3)  # best on 2026Feb8
        else:
            raise ValueError(f"Unknown material: {self.material}")


# -----------------------------
# Training / supervisor
# -----------------------------
@dataclass(frozen=True)
class TrainingConfig:
    T: int = 4  # total training set time (not time to reach equilibrium during every step)

    # desired_buckle_type: str = 'random'
    # desired_buckle_type: str = 'opposite'
    # desired_buckle_type: str = 'straight'
    desired_buckle_type: str = 'specified'

    if desired_buckle_type == 'random':
        desired_buckle_rand_key: int = 169  # key for seed of random sampling of buckle pattern
    elif desired_buckle_type == 'specified':
        desired_buckle_pattern: tuple = (-1, -1, -1, -1, -1)  # desired buckle, 1=up
        # desired_buckle_pattern: tuple = (-1, -1, -1, -1)  # desired buckle, 1=up
        # desired_buckle_pattern: tuple = (1, -1, -1, -1, -1, 1, 1, 1)  # desired buckle, 1=up

    init_buckle_pattern: tuple = (-1, 1, -1, -1, -1)  # initial buckle, 1=up
    # init_buckle_pattern: tuple = (-1, -1, -1, 1)  # initial buckle, 1=up
    # init_buckle_pattern: tuple = (-1, -1, 1, 1, 1, 1, -1, -1)  # initial buckle, 1=up
    # init_buckle_pattern: tuple = (1)  # initial buckle, 1=up

    # dataset_sampling: str = 'uniform'  # random uniform vals for x, y, angle
    # dataset_sampling: str = 'predetermined'  # import measured F along predetermined trajectory every training step t
    dataset_sampling: str = 'free_tip'  # free tip pos and angle (zero forces at sensor) during measurement
    # dataset_sampling: str = 'specified'  # constant
    # dataset_sampling: str = 'tile'  # constant
    # dataset_sampling = 'almost_flat'  # flat piece w a bit of constant noise, single measurement
    # dataset_sampling = 'flat'  # flat piece, single measurement
    # dataset_sampling = 'stress strain'

    predet_traj_file: str = r"Predetermined trajectory\June15\example_traj.csv"

    # dataset_file: str = r"Predetermined trajectory\Mar23\buckle={}.csv"
    # dataset_file: str = r"Predetermined trajectory\May27\short_arc\May24Chain_1stEnd\buckle={}.csv"  # 2026June7
    dataset_file: str = r"Predetermined trajectory\June15\1stEnd\buckle={}.csv"  # 2026June7

    # # tip values to buckle shims - 'BEASTAL' for the BEASTAL scheme, else 'one_to_one'
    # update_scheme: str = 'one_to_one'  # direct normalized loss, equal to num of outputs
    # update_scheme: str = 'loss_diff'  # difference of x and y loss components
    # update_scheme: str = 'loss_x_trend'  # delta tip by trend of loss x, delta angle by addition of losses
    update_scheme: str = 'pos'  # difference in measured and desired tip position and angle
    pos_delta_mode: str = 'signed'  # 'signed' = Mar23 sign factors, 'direct' = May3 direct position loss
    use_tangent_clamp: bool = True  # If True, preserve tangential update direction when clamping the free tip.
    # update_scheme: str = 'lossx_concavity'  # tip_angle changes due to concavity of loss x along trajectory.
    #                                         # tip pos due to loss_y sign
    # update_scheme: str = 'radial_one_to_one'  # evolve tip angle and large radius due to instantaneous loss
    # update_scheme: str = 'BEASTAL'  # update using the BEASTAL scheme (with pseudoinverse of the incidence matrix).
    # update_scheme: str = 'BEASTAL_no_pinv'  # update using (y_j)(Loss_j), no psuedo inv of the incidence matrix.
    # update_scheme: str = 'radial_halfway_BEASTAL'  # evolve tip angle and large radius due to instantaneous loss
    # update_scheme: str = 'radial_BEASTAL'  # update using BEASTAL (pseudoinverse of 2x2 incidence matrix),
                                             # calculated in total and tip angles

    # normalize_step: bool = True
    normalize_step: bool = False

    if (update_scheme == 'radial_BEASTAL' or update_scheme == 'pos') and not normalize_step:
        alpha: float = 0.25  # learning rate
    elif normalize_step:
        alpha = 0.25
    else:
        # alpha = 0.2  # learning rate  #  Apr23
        if HINGES < 6:
            alpha = 0.15  # learning rate # Apr30
        else:
            alpha = 0.35

    if update_scheme == 'pos' and HINGES > 2:
        tradeoff_pos_angle: float = 1/2 + 1/8 * HINGES
    else:
        tradeoff_pos_angle = 1

    # control_tip: bool = True  # imposed tip position in measurement and update. If False, tip is free
    control_tip: bool = False  # imposed tip position in measurement and update. If False, tip is free
    control_first_edge: bool = True  # if True, fix nodes (0, 1) to zero. if Flase, just the first

    rand_key_dataset: int = 7  # for random sampling of dataset, if dataset_sampling is True
    rand_key_tip: int = 8  # for random sampling of update tip positions, once forces explode

    convert_pos = 1000  # convert [m] to [mm]
    convert_angle = 180/np.pi  # convert rad to deg
    convert_F = 1  # already in [mN]


# -----------------------------
# Top-level config
# -----------------------------
@dataclass(frozen=True)
class ExperimentConfig:
    Strctr: StructureConfig = StructureConfig()
    Variabs: VariablesConfig = VariablesConfig()
    Eq: EquilibriumConfig = EquilibriumConfig()
    Train: TrainingConfig = TrainingConfig()


CFG = ExperimentConfig()
