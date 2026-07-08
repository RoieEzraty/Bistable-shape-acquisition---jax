from __future__ import annotations

import numpy as np
import copy
import jax
import jax.numpy as jnp
import equinox as eqx
from jax import vmap
from datetime import datetime

from typing import Tuple, List
from numpy import array, zeros
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional

import helpers_builders, learning_funcs

if TYPE_CHECKING:
    from StructureClass import StructureClass
    from StateClass import StateClass
    from EquilibriumClass import EquilibriumClass
    from VariablesClass import VariablesClass

np.set_printoptions(precision=4, suppress=True)


def _local_log_time() -> str:
    """Return a readable local timestamp for sparse event logs."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ===================================================
# Class - Supervisor Variables - training set, losses, etc.
# ===================================================


class SupervisorClass:
    """
    Variables that are by the external supervisor in the experiment

    attributes:
    -----------
    alpha              : float, Step size for updating the commanded tip pose.
    T                  : int, Number of training steps in the dataset.
    desired_buckle_arr : ndarray[int] (H,S), desired buckle configuration.
    desired_pos_in_t   : ndarray (nodes, 2, T), whole chain configuration in desired buckle state, for every training measurement,
                         not used for learning, just for forces.
    desired_Fx/Fy_in_t : ndarray (T,), forces sensed in the desired buckle configuration for every training step measurement
    tip_pos_in_t       : ndarray (T, 2), training dataset tip positions, Measurement modality.
    tip_angle_in_t     : ndarray (T,), training dataset tip angles, Measurement modality.
    loss_in_t          : ndarray (T, loss_dim), loss (x and y, optionally angle) for every measurement during training.
    loss_MSE_in_t      : ndarray (T,), Mean Squared Error of the x,y loss.
    Hamming_distance_in_t : ndarray (T,), Hamming distance between measured and desired buckles.
    tip_pos_update_in_t: ndarray (T, 2), tip position in the Update modality, for every training step.
    tip_angle_update_in_t : ndarray (T, ), tip angle in the Update modality, for every training step.
    total_angle_update_in_t : ndarray (T, ), angle between tip and end of first link, in Update modality, for every training step
    imposed_mask       : # (2*nodes,), boolean of whether a node (ends of edges/facets) is imposed or not
                         True only at two final nodes, if control_tip==True
    loss               : (2,)/(3,), instantaneous loss (forces/position)
    control_tip        : bool, default=True, control tip position and angle. If False, release tip, chain is free at end.
    control_first_edge : bool, default=True, nodes 0 and 1 are fixed.  If False, only node 0 is fixed.
    normalize_step     : bool, default=True. normalize Update position and angle step size so won't be too large or small
    update_scheme      : str, How tip commands are updated from the loss:
                         'one_to_one'      = direct normalized loss, equal to num of outputs
                         'BEASTAL'         = update using pseudoinverse of the incidence matrix.
                         'BEASTAL_no_pinv' = update using (y_j)(Loss_j), no psuedo inv of the incidence matrix.
    R_free             : Maximal allowed radius [mm] of a taut chain from end of 1st link to beginning of last, up to some margin.
                         To correct for stretch, tip position never surpasses it.
    convert_pos        : conversion scale from [m] to [mm], for file exports
    convert_angle      : conversion scale from [rad] to [deg], for file exports
    convert_F          : coversion scale of forces needs no adjustment, it is in [mN], for file exports.

    Methods:
    --------
    _build_imposed_mask(Strctr, control_tip)
            Boolean mask marking imposed (prescribed) degrees of freedom.
            These are prescribed position, generally tip control.
    create_dataset(Strctr, CFG, sampling, tip_pos, tip_angle, dist_noise, angle_noise)
            Generate and store commanded tip positions and angles for the supervisor.
            according to sampling strategy. These trajectories are used in measurement, update, or stress–strain protocols.
    set_desired(pos_arr, Fx, Fy, t):
            Store ground-truth targets for step t.
    calc_loss(Variabs, t, Fx, Fy)
            Compute loss vector (Fx,Fy) at step t and log it.
    calc_update_tip(t, Strctr, Variabs, State, current_tip_pos, prev_tip_update_pos, current_tip_angle,
                    prev_tip_update_angle, correct_for_total_angle, correct_for_coil, correct_for_cut_origin
            Compute next tip position/angle commands from current loss and state (pure NumPy).
    """
    # --- configuration / hyperparams ---
    T: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    update_scheme: str = eqx.field(static=True)
    pos_delta_mode: str = eqx.field(static=True)
    use_tangent_clamp: bool = eqx.field(static=True)
    control_tip: bool = eqx.field(static=True)
    control_first_edge: bool = eqx.field(static=True)
    normalize_step: bool = eqx.field(static=True)
    R_free: float = eqx.field(static=True)
    R_min: float = eqx.field(static=True)
    convert_pos: float = 1000  # convert [m] to [mm]
    convert_angle: float = 180/np.pi  # convert rad to deg
    convert_F: float = 1  # already in [mN]

    # --- desired targets (fixed-size buffers; NumPy, mutable at runtime) ---
    desired_buckle_arr: NDArray[np.int32] = eqx.field(static=True)                 # (hinges,)
    desired_pos_in_t: NDArray[np.float32] = eqx.field(init=False, static=True)     # (nodes, 2, T)
    desired_Fx_in_t: NDArray[np.float32] = eqx.field(init=False, static=True)      # (T,)
    desired_Fy_in_t: NDArray[np.float32] = eqx.field(init=False, static=True)      # (T,)

    # --- dataset inputs (what tip we command at each step) ---
    tip_pos_in_t: NDArray[np.float32] = eqx.field(init=False, static=True)         # (T, 2)
    tip_angle_in_t: NDArray[np.float32] = eqx.field(default=None, init=False, static=True)    # (T,)

    # --- running logs / losses ---
    loss_in_t: NDArray[np.float32] = eqx.field(init=False, static=True)            # (T, 2)/(T, 3)
    loss_MSE_in_t: NDArray[np.float32] = eqx.field(init=False, static=True)        # (T,)
    Hamming_distance_in_t: NDArray[np.int32] = eqx.field(init=False, static=True)  # (T,)
    tip_pos_update_in_t: NDArray[np.float32] = eqx.field(init=False, static=True)  # (T, 2)
    tip_angle_update_in_t: NDArray[np.float32] = eqx.field(default=None, init=False, static=True)  # (T,)
    total_angle_update_in_t: NDArray[np.float32] = eqx.field(default=None, init=False, static=True)  # (T,)

    # ------ for equilibrium calculation, jax arrays ------
    # imposed_mask: jax.ndarray[bool] = eqx.field(static=True)                     # (2*nodes,)
    imposed_mask_w_tip: jax.ndarray[bool] = eqx.field(static=True)                 # (2*nodes,)
    imposed_mask_free: jax.ndarray[bool] = eqx.field(static=True)                  # (2*nodes,)

    # --- scratch (most recent loss vector) ---
    loss: NDArray[np.float32] = eqx.field(init=False, static=True)                 # (2,)

    def __init__(self, Strctr, CFG, supress_prints: bool = True) -> None:
        self.T = int(CFG.Train.T)  # total training-set size (& algorithm time, dont confuse with time to equilib state)
        self.alpha = float(CFG.Train.alpha)
        self.tradeoff_pos_angle = float(CFG.Train.tradeoff_pos_angle)
        self.update_scheme = str(CFG.Train.update_scheme)
        self.pos_delta_mode = str(getattr(CFG.Train, "pos_delta_mode", "signed double"))
        if self.pos_delta_mode not in {"signed", "signed double", "direct"}:
            raise ValueError(f"Unknown pos_delta_mode='{self.pos_delta_mode}'")
        self.use_tangent_clamp = bool(getattr(CFG.Train, "use_tangent_clamp", True))
        self.control_tip = bool(CFG.Train.control_tip)
        self.control_first_edge = bool(CFG.Train.control_first_edge)  # if true, fix nodes (0, 1), else fix only (0)

        self.stop_if_symmetrical = bool(CFG.Train.stop_if_symmetrical)  # stop training if chain in state symmetrical to desired

        # for equilibrium
        self.imposed_mask_w_tip = self._build_imposed_mask(Strctr, control_tip=True)
        self.imposed_mask_free = self._build_imposed_mask(Strctr, control_tip=False)

        # Desired/targets
        if CFG.Train.desired_buckle_type == 'random':  # uniformly distributed values of +1 and -1
            key = jax.random.PRNGKey(CFG.Train.desired_buckle_rand_key)   # seed
            desired_buckle = jax.random.randint(key, (Strctr.hinges, Strctr.shims), minval=-1, maxval=2)  # +1, 0 or -1
            desired_buckle = desired_buckle.at[desired_buckle == 0].set(-1)  # replace 0 w/ -1
        elif CFG.Train.desired_buckle_type == 'opposite':  # opposite than init buckle, requires creating initial buckle
            desired_buckle = - helpers_builders._initiate_buckle(Strctr.hinges, Strctr.shims,
                                                                 buckle_pattern=CFG.Train.init_buckle_pattern,
                                                                 numpify=True)
        elif CFG.Train.desired_buckle_type == 'straight':  # same as initial buckle, requires creating initial buckle
            desired_buckle = helpers_builders._initiate_buckle(Strctr.hinges, Strctr.shims,
                                                               buckle_pattern=CFG.Train.init_buckle_pattern,
                                                               numpify=True)
        elif CFG.Train.desired_buckle_type == 'specified':
            desired_buckle = helpers_builders._initiate_buckle(Strctr.hinges, Strctr.shims,
                                                               buckle_pattern=CFG.Train.desired_buckle_pattern,
                                                               numpify=True)
        self.desired_buckle_arr = np.asarray(desired_buckle, dtype=np.int32)
        self.desired_pos_in_t = zeros((Strctr.nodes, 2, self.T), dtype=np.float32)
        self.desired_Fx_in_t = zeros((self.T), dtype=np.float32)
        self.desired_Fy_in_t = zeros((self.T), dtype=np.float32)

        # Dataset (commands)
        self.tip_pos_in_t = zeros((self.T, 2), dtype=np.float32)
        self.tip_angle_in_t = zeros((self.T,), dtype=np.float32)

        # Logs / updates
        if self.update_scheme == 'pos':
            loss_size = 3
        else:
            loss_size = 2
        self.loss_in_t = zeros((self.T, loss_size), dtype=np.float32)
        self.loss_MSE_in_t = zeros((self.T,), dtype=np.float32)
        self.Hamming_distance_in_t = zeros((self.T,), dtype=np.int32)
        self.Hamming_distance = 0

        # Last loss vector (shape matches control mode)
        self.loss = zeros(loss_size, dtype=np.float32)

        self.tip_pos_update_in_t = zeros((self.T, 2), dtype=np.float32)
        self.tip_angle_update_in_t = zeros((self.T,), dtype=np.float32)
        self.total_angle_update_in_t = zeros((self.T,), dtype=np.float32)

        self.normalize_step = bool(CFG.Train.normalize_step)  # whether to normalize train step in [x, y, theta] space

        self.R_free = (Strctr.edges - 2*0.98)*Strctr.L  # maximal radius the chain could have, up to some margin
        self.R_min = Strctr.L  # minimal radius around [L/2, 0] that tip cannot cross

        # for output files
        self.convert_pos = CFG.Train.convert_pos
        self.convert_angle = CFG.Train.convert_angle
        self.convert_F = CFG.Train.convert_F

        # invert tip changes if training fails
        self.invert_delta_tip = False

        # chain in structure that is symmetrical to desired
        self.symmetrical_state = False

        # wrap angle while accounting for shortening of tip
        if Strctr.hinges < 5:
            self.wrap = True
        else:
            self.wrap = False

        # tip restart bookkeeping for origin-cut handling
        self.origin_cut_restart_count = 0  # consecutive origin-cut restarts
        self.coil_count = 0  # consecutive restarts due to tip coiling
        self.last_restart_reason: Optional[str] = None  # None | "origin_cut" | "coil"
        self.origin_restart_base_frac = 0.6  # base vertical offset in units of L
        self.origin_restart_step_frac = 0.6  # extra offset per repeated cut, in units of L

        self.rng_tip = np.random.default_rng(CFG.Train.rand_key_tip)

        # prints during run if True
        self.supress_prints = supress_prints

    # ---------------------------------------------------------------
    # Imposed mask boolean
    # ---------------------------------------------------------------
    def _build_imposed_mask(self, Strctr: "StructureClass", control_tip: bool = True) -> jax.Array:
        """
        Boolean mask marking imposed (prescribed) degrees of freedom. These are prescribed position, generally tip control.

        Parameters
        ----------
        Strctr      : StructureClass Structural definition containing:
                      nodes    : number of nodes (H+2)
                      n_coords : total number of coordinates (= 2 * nodes)
        control_tip : bool, default=True. if True, tip node and the one immediately before it are imposed. Else free tip

        Returns
        -------
        imposed_mask : jnp.ndarray, shape (2 * nodes,), dtype=bool, [x0, y0, x1, y1, ..., x_last, y_last]
        """
        n_coords = Strctr.n_coords  # 2 * nodes
        N = Strctr.nodes  # number of nodes
        last = N - 1

        # --- fixed and imposed DOFs initialize ---
        imposed_mask = jnp.zeros((n_coords,), dtype=bool)

        # -------- imposed tip position ----------
        if control_tip:
            # set tip indices as true
            idxs = jnp.array([helpers_builders.dof_idx(last, 0), helpers_builders.dof_idx(last, 1)])
            before_last_idxs = jnp.array([helpers_builders.dof_idx(last - 1, 0), helpers_builders.dof_idx(last - 1, 1)])
            idxs = jnp.concatenate([before_last_idxs, idxs])
            imposed_mask = imposed_mask.at[idxs].set(True)
        return imposed_mask

    # ---------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------
    def create_dataset(self, Strctr: "StructureClass", CFG, sampling: str, tip_pos: Optional[NDArray] = None,
                       tip_angle: Optional[float] = None, dist_noise: float = 0.01, angle_noise: float = 0.1) -> None:
        """
        Generate and store commanded tip positions and angles for the supervisor.
        according to sampling strategy. These are used in measurement, update, or stress–strain protocols.

        Parameters
        ----------
        Strctr     : StructureClass
        CFG        : ExperimentConfig. Uses:
                     CFG.Train.rand_key_dataset  - random seed for reproducible datasets
                     CFG.Variabs.exp_start       - start position for ADMET stress–strain from Harvard
                     CFG.Variabs.distance        - compression distance for stress–strain
        sampling   : str. Dataset generation mode. One of:
                    "uniform": Random uniform sampling in a bounded box:
                               x ∈ [(edges - 1.5)L , (edges - 0.5)L]
                               y ∈ [-L/2 , L/2]
                               θ ∈ [-π/5 , π/5]
                               Uses numpy Generator seeded with CFG.Train.rand_key_dataset.
                    "flat": Fully flat configuration over all T:
                            tip_pos = [edges, 0]
                            tip_angle = 0
                    "almost flat": Tip is placed slightly compressed relative to flat state.
                    "specified": User-provided fixed (tip_pos, tip_angle) repeated over T.
                    "stress strain": compression–decompression trajectory along x-axis:
                                     start → end → start over T steps (triangular waveform), for ADMET Harvard experiment
                                     Optional:
                                     dist_noise  - constant y-offset
                                     angle_noise - constant angle offset
                     "tile" - Repeats blocks of tip_pos and tip_angle to fill T.
        tip_pos     : ndarray of shape (2,), optional, only for "specified" or "tile"
        tip_angle   : float, optional, optional, only for "specified" or "tile"
        dist_noise  : float, default 0.0. Only for "stress strain". Constant y-offset.
        angle_noise : float, default 0.0. Only for "stress strain".

        Returns:
        --------
        self.tip_pos_in_t   : (T, 2 )tip position [mm] in dataset, not Update
        self.tip_angle_in_t : (T,) tip angle [rad] in dataset, not Update
        """
        # save as variable
        self.dataset_sampling = sampling
        if sampling == 'predetermined':
            self.dataset_file = CFG.Train.dataset_file
            self.predet_traj_file = CFG.Train.predet_traj_file

        # tip positions and angles for specified tip dataset
        if sampling == 'uniform':
            rng = np.random.default_rng(CFG.Train.rand_key_dataset)
            low = array([(Strctr.edges - 1.0) * Strctr.L, -Strctr.L * 2 / 3, -np.pi / 5])  # lowest allowed value
            high = array([(Strctr.edges - 0.4) * Strctr.L, Strctr.L * 2 / 3, np.pi / 5])  # highest allowed value
            # low = array([(Strctr.edges - 0.5) * Strctr.L, -Strctr.L * 1 / 3, -np.pi / 5])  # lowest allowed value
            # high = array([(Strctr.edges - 0.01) * Strctr.L, Strctr.L * 1 / 3, np.pi / 5])  # highest allowed value
            samples = rng.uniform(low, high, size=(self.T, 3)).astype(np.float32)  # (T, 3) sample size
            self.tip_pos_in_t = samples[:, :2]  # (T, 2)
            self.tip_angle_in_t = samples[:, 2]  # (T,)

            # correct for too big stretch during measurement
            # ------ clamp overstretched dataset samples ------
            for t in range(self.T):
                self.tip_pos_in_t[t, :] = helpers_builders._correct_big_stretch(tip_pos=self.tip_pos_in_t[t, :],
                                                                                tip_angle=float(self.tip_angle_in_t[t]),
                                                                                total_angle=0.0, R_free=self.R_free,
                                                                                L=Strctr.L, margin=0.1, wrap=self.wrap,
                                                                                supress_prints=self.supress_prints)
        elif sampling in {'flat', 'almost_flat', 'specified', 'predetermined', 'free_tip'}:
            end = float(Strctr.edges*Strctr.L)
            if sampling in {'flat', 'predetermined', 'free_tip'}:
                tip_pos = array([end, 0], dtype=np.float32)
                tip_angle = 0.0
            elif sampling == 'almost_flat':
                tip_pos = array([end-dist_noise, +dist_noise], dtype=np.float32)  # flat arrangement
                tip_angle = angle_noise
            else:  # == 'specified'
                pass
            self.tip_pos_in_t[:] = np.tile(tip_pos, (self.T, 1))
            self.tip_angle_in_t[:] = np.tile(tip_angle, (self.T, ))
        elif sampling == 'stress strain':
            start = 2*Strctr.L + CFG.Variabs.exp_start
            end = start - CFG.Variabs.distance
            tip_in = np.linspace(start, end, self.T // 2, endpoint=False)  # decreasing: start -> end
            tip_out = np.linspace(end, start, self.T - self.T // 2, endpoint=False)  # increasing: end -> start
            tip_arr = np.concatenate([tip_in, tip_out])  # shape (self.T,),  back-and-forth trajectory

            noisy_zeros_arr = np.zeros_like(tip_arr) + dist_noise  # shape (T,)
            self.tip_pos_in_t[:] = np.column_stack((tip_arr, noisy_zeros_arr))  # shape (T, 2)
            self.tip_angle_in_t[:] = angle_noise
        elif sampling == 'tile':
            self.tip_pos_in_t[:] = np.tile(tip_pos, (self.T // len(tip_pos) + 1, 1))[:self.T]
            tip_angles_block = np.repeat(tip_angle, tip_pos.shape[0])
            self.tip_angle_in_t[:] = np.tile(tip_angles_block, self.T // len(tip_angles_block) + 1)[:self.T]
        else:
            raise ValueError(f"Incompatible sampling='{sampling}'")

    def set_desired(self, pos_arr: jax.Array, Fx: float, Fy: float, t: int) -> None:
        """Store ground-truth targets for step t.

        Parameters:
        -----------
        pos_arr : (N, 2) node positions in x and y [m]
        Fx      : float force in global x direction [mN]
        Fy      : float force in global y direction [mN]
        t       : {0:self.T} current time step [dimless]
        """
        self.desired_pos_in_t[:, :, t] = helpers_builders.jax2numpy(pos_arr)  # [m]
        self.desired_Fx_in_t[t] = float(Fx)  # [mN]
        self.desired_Fy_in_t[t] = float(Fy)  # [mN]

    def calc_Hamming_distance(self, measured_buckle_arr: NDArray, t: Optional[int] = None) -> int:
        """Calculate and store the Hamming distance from measured to desired buckles.

        Parameters:
        -----------
        measured_buckle_arr : ndarray, shape (H,S)
            Current measured buckle configuration.
        t : Optional[int]
            Current time step. If supplied, the distance is also stored in the time history.
        """
        measured = np.asarray(measured_buckle_arr)
        desired = np.asarray(self.desired_buckle_arr)
        if measured.shape != desired.shape:
            raise ValueError(f"measured_buckle_arr shape {measured.shape} does not match desired {desired.shape}")

        self.Hamming_distance = int(np.count_nonzero(measured != desired))
        if t is not None:
            self.Hamming_distance_in_t[t] = self.Hamming_distance
        return self.Hamming_distance

    # ---------------------------------------------------------------
    # Calculations - loss and Update values
    # ---------------------------------------------------------------
    def calc_loss(self, Variabs: "VariablesClass", t: int, Fx: Optional[float] = None, Fy: Optional[float] = None,
                  pos: Optional[NDArray] = None, pos_des: Optional[NDArray] = None,
                  Strctr: Optional["StructureClass"] = None,
                  thresh_for_symmetrical=10**(-3)) -> None:
        """Compute loss vector (Fx,Fy) at step t and log it.

        Parameters:
        -----------
        Variabs : VariablesClass, using:
                  - norm_force: float typical force calculated in Variabs.init
        t       : {0:self.T} current time step
        Fx      : float force in global x direction [mN]
        Fy      : float force in global y direction [mN]
        Strctr  : StructureClass, optional. Used for position-loss tip angle via accumulated hinge angles.

        Returns:
        --------
        loss     - float, F_hat-F in 2d
        loss_MSE - float, mean squared loss
        """
        dispatch = self._get_loss_dispatch()
        if self.update_scheme == 'pos':
            fn = dispatch.get("position", None)
        else:
            fn = dispatch.get("force", None)
        self.loss = fn(Variabs, t, Fx, Fy, pos, pos_des, Strctr)

        # normalize loss
        # self.loss = self.loss / Variabs.norm_force  # [dimless]

        # put in loss vec
        self.loss_in_t[t, :self.loss.shape[0]] = self.loss

        if (pos is not None) and (pos_des is not None) and self.stop_if_symmetrical:
            is_symmetrical = helpers_builders.symmetrical_chain_state(pos, pos_des, thresh_for_symmetrical)
        else:
            is_symmetrical = False

        if is_symmetrical:
            print('symmetrical chain state, invert chain, training is done')
            self.symmetrical_state = True
            self.loss_MSE = 0.0
        else:
            self.loss_MSE = np.mean(self.loss**2)
        self.loss_MSE_in_t[t] = self.loss_MSE

    def calc_concavity(self, F_meas_full_traj, F_des_full_traj) -> None:
        """
        """
        loss_x_full = F_des_full_traj[:, 0] - F_meas_full_traj[:, 0]
        middle = int(np.floor(len(loss_x_full)/2))
        abs_mean_loss = np.mean(np.abs(loss_x_full))
        if abs_mean_loss < 10**(-6):  # if loss is zero
            self.concavity = 0.0
        else:
            self.concavity = -((loss_x_full[0] + loss_x_full[-1]) -
                               (loss_x_full[middle-1] + loss_x_full[middle])) / (2 * abs_mean_loss)

    def calc_loss_x_trend(self, F_meas_full_traj, F_des_full_traj) -> None:
        """
        Calculate the linear trend/slope of Fx_desired - Fx_measured
        along the full predetermined trajectory.
        """
        loss_x_full = F_des_full_traj[:, 0] - F_meas_full_traj[:, 0]

        if len(loss_x_full) < 2 or np.mean(np.abs(loss_x_full)) < 1e-6:
            self.loss_x_trend = 0.0
        else:
            traj_idx = np.arange(len(loss_x_full), dtype=float)
            self.loss_x_trend = float(np.polyfit(traj_idx, loss_x_full, deg=1)[0])

    def calc_update_tip(self, t: int, Strctr: "StructureClass", Variabs: "VariablesClass",
                        State_meas: "StateClass", State_update: "StateClass",
                        correct_for_total_angle: Optional[bool] = False, correct_for_coil: Optional[bool] = True,
                        correct_for_cut_origin: Optional[bool] = True,
                        correct_for_update_force: Optional[bool] = True) -> None:
        """Compute next tip position/angle commands from current loss and state (pure NumPy).

        Parameters:
        -----------
        t
        correct_for_... : booleans, whether to correct tip pos due to: addition to total tip angle relative to origin,
                                                                       coiled tip (reset tip values from dataset)
                                                                       tip cuts origin (as above)
                                                                       forces at tip exceed threshold

        Returns:
        --------
        updates in self:
        np.array(float), (2,) update_tip_pos_in_t [m]
        float, tip_angle_update_in_t [mN]
        float, total_angle_update_in_t [mN]
        """
        # ------ delta tip and angle ------
        dispatch = self._get_delta_dispatch()
        fn = dispatch.get(self.update_scheme, None)  # function to calculate delta tip and angle from update_scheme
        if fn is None:
            raise ValueError(f"Unknown update_scheme='{self.update_scheme}'")
        delta_tip_x, delta_tip_y, delta_angle = fn(t, Strctr, Variabs, State_meas)
        delta_tip = array([delta_tip_x, delta_tip_y])  # assemble into 3d array
        # multiply space in factor when number of hinges is large
        delta_tip *= self.tradeoff_pos_angle
        if not self.supress_prints:
            print(f'delta_tip before corr {delta_tip}')
            print(f'delta_angle before corr {delta_angle}')

        # ------ normalize step if update is non-zero ------
        if self.normalize_step and np.linalg.norm(np.append(delta_tip, delta_angle)) > 10**(-12):
            # old version up to Feb22
            step_size = np.linalg.norm(np.append(delta_tip, delta_angle))
            # print(f'step_size={step_size}')
            if self.update_scheme == 'loss_diff':
                tradeoff_pos_angle = 1/4 * Strctr.hinges  # generalize to many hinges. with 4 tradeoff should be 1
            else:
                # tradeoff_pos_angle = 1/2
                tradeoff_pos_angle = 2
            delta_tip = copy.copy(delta_tip) / step_size * self.alpha
            delta_angle = copy.copy(delta_angle) / step_size * self.alpha * tradeoff_pos_angle

            if not self.supress_prints:
                print(f'normalized position to {delta_tip}')
                print(f'normalized angle to {float(delta_angle)}')

        # ------ insert into vectors in time ------
        # get previous tip positions
        if t == 1:
            prev_tip_update_pos = self.tip_pos_in_t[t, :]
            prev_tip_update_angle = self.tip_angle_in_t[t]
        else:
            prev_tip_update_pos = self.tip_pos_update_in_t[t - 1, :]
            prev_tip_update_angle = self.tip_angle_update_in_t[t - 1]
        if not self.supress_prints:
            print(f'prev_tip_update_pos{prev_tip_update_pos}')
            print(f'prev_tip_update_angle{prev_tip_update_angle}')

        # invert delta_tip if required
        if self.invert_delta_tip is True:  # invert tip
            delta_tip = -delta_tip
            delta_angle = -delta_angle

        # add to tip update in time
        self.tip_pos_update_in_t[t, :] = prev_tip_update_pos + delta_tip
        self.tip_angle_update_in_t[t] = prev_tip_update_angle + float(delta_angle)

        # ------ correct for total angle ------
        # add change in tip angle to the total angle from the origin
        if correct_for_total_angle:
            if t == 1:
                prev_total_angle = 0.0
            else:
                prev_total_angle = self.total_angle_update_in_t[t - 1]
            total_angle: float = helpers_builders._get_total_angle(self.tip_pos_update_in_t[t, :], prev_total_angle,
                                                                   Strctr.L)
            self.total_angle_update_in_t[t] = total_angle
            delta_total_angle: float = total_angle - prev_total_angle
            self.tip_angle_update_in_t[t] += delta_total_angle
            if not self.supress_prints:
                print(f'total angle {total_angle}')
                print(f'add delta tip angle {delta_total_angle} to correct for total angle ')

        # ------ correct for to big a stretch ------
        if self.dataset_sampling == 'free_tip' and self.update_scheme == 'pos':
            # If the raw x/y update exits the reachable disk, slide along the
            # effective-radius perimeter instead of radially projecting back.
            R_eff = helpers_builders.effective_radius(self.R_free, Strctr.L, total_angle=total_angle,
                                                      tip_angle=float(self.tip_angle_update_in_t[t]),
                                                      supress_prints=self.supress_prints)
            clamp_margin = 0.1 * Strctr.L * (Strctr.hinges-1)
            before_prev = helpers_builders._get_before_tip(prev_tip_update_pos, float(prev_tip_update_angle),
                                                           Strctr.L, xp=np)

            # print("raw tip", self.tip_pos_update_in_t[t, :])
            # print("u", np.array([np.cos(self.tip_angle_update_in_t[t]), np.sin(self.tip_angle_update_in_t[t])]))
            # print("before_prev", before_prev)
            # print("before_raw", self.tip_pos_update_in_t[t, :] - Strctr.L*np.array([
            #     np.cos(self.tip_angle_update_in_t[t]),
            #     np.sin(self.tip_angle_update_in_t[t])
            # ]))
            # print("R_eff", R_eff)
            # print("total_angle", total_angle, "tip_angle", self.tip_angle_update_in_t[t])
            # # clamp outside inner radius
            tip_new, _, clamped_inner = helpers_builders.clamp_pos_same_delta(before_prev=before_prev,
                                                                              tip_angle_new=float(self.tip_angle_update_in_t[t]),
                                                                              tip_raw=self.tip_pos_update_in_t[t, :],
                                                                              second_node=array([Strctr.L, 0.0], dtype=float),
                                                                              R_lim=self.R_min, L=Strctr.L, mod="inner",
                                                                              clamp_margin=clamp_margin)

            # clamp outside outer radius
            # tip_new = helpers_builders._correct_big_stretch(tip_new, self.tip_angle_update_in_t[t], total_angle,
            #                                                 self.R_free, Strctr.L, margin=0.0,
            #                                                 supress_prints=self.supress_prints)
            tip_new, _, clamped_outer = helpers_builders.clamp_pos_same_delta(before_prev=before_prev,
                                                                              tip_angle_new=float(self.tip_angle_update_in_t[t]),
                                                                              tip_raw=tip_new,
                                                                              second_node=array([Strctr.L, 0.0], dtype=float),
                                                                              R_lim=R_eff, L=Strctr.L, mod="outer",
                                                                              tip_update_prev=prev_tip_update_pos,
                                                                              raw_update_tip=delta_tip, clamp_margin=clamp_margin,
                                                                              use_tangent_selection=self.use_tangent_clamp)

            if clamped_outer:
                corrected_delta = tip_new - prev_tip_update_pos
                # New 2026June25
                eps = max(1e-12, 1e-6 * Strctr.L)
                clamp_reversed_x = (abs(delta_tip[0]) > eps
                                    and np.sign(corrected_delta[0]) != np.sign(delta_tip[0]))
                clamp_fabricated_x = (abs(delta_tip[0]) <= eps
                                      and abs(corrected_delta[0]) > abs(corrected_delta[1]) + eps)
                clamp_erased_y = abs(delta_tip[1]) > eps and abs(corrected_delta[1]) < 0.1 * abs(delta_tip[1])
                two_step_bounce = (t > 2
                                   and np.linalg.norm(tip_new - self.tip_pos_update_in_t[t - 2, :]) < eps)
                if (clamp_erased_y and (clamp_reversed_x or clamp_fabricated_x)) or two_step_bounce:
                    tip_new = prev_tip_update_pos.copy()
                    corrected_delta = tip_new - prev_tip_update_pos
                print(f"[{_local_log_time()}] outer clamp:",
                      "t=", t,
                      "raw_delta=", delta_tip,
                      "corrected_delta=", corrected_delta,
                      "raw_dy=", delta_tip[1],
                      "corrected_dy=", corrected_delta[1],
                      "prev_y=", prev_tip_update_pos[1],
                      "R_eff=", R_eff,
                      "clamp_margin=", clamp_margin)

            self.tip_pos_update_in_t[t, :] = tip_new

            if not self.supress_prints:
                # if clamped_outer:
                #     print(f'tip slid on effective outer radius to {self.tip_pos_update_in_t[t, :]}')
                if clamped_inner:
                    print(f'tip slid on effective inner radius to {self.tip_pos_update_in_t[t, :]}')

        else:
            self.tip_pos_update_in_t[t, :] = helpers_builders._correct_big_stretch(self.tip_pos_update_in_t[t],
                                                                                   self.tip_angle_update_in_t[t],
                                                                                   total_angle, self.R_free, Strctr.L,
                                                                                   margin=0.1,
                                                                                   supress_prints=self.supress_prints)
            if not self.supress_prints:
                print(f'tip after correct big stretch={self.tip_pos_update_in_t[t, :]}')

        # ------ correct for coil or cut origin ------
        cond_coil = helpers_builders.coil(self.tip_angle_update_in_t[t], revolutions=0.375*Strctr.hinges)

        cond_cut_origin = helpers_builders.swept_last_edge_crosses_first_edge(tip_prev=prev_tip_update_pos,
                                                                              angle_prev=prev_tip_update_angle,
                                                                              tip_new=self.tip_pos_update_in_t[t, :],
                                                                              angle_new=self.tip_angle_update_in_t[t],
                                                                              L=Strctr.L, include_endpoints=False)

        cond_tip_force = helpers_builders.tip_force(State_update.Fx, State_update.Fy, Variabs.norm_force)

        self.restart = False

        if correct_for_cut_origin and cond_cut_origin:
            print(f'[{_local_log_time()}] origin is cut at t={t}')
            self.coil_count = 0
            self.origin_cut_restart_count += 1

            # # from below -> restart slightly below, from above -> slightly above
            # side_sign = helpers_builders._origin_cut_side(before_prev=before_tip_tminus1,
            #                                               tip_prev=self.tip_pos_update_in_t[t-1, :],
            #                                               before_new=before_tip_t,
            #                                               tip_new=self.tip_pos_update_in_t[t, :])

            # sign is just from angle direction of tip
            side_sign = np.sign(delta_angle)
            self._restart_flat_with_y_bias(t, Strctr, side_sign=side_sign)
            print(f'[{_local_log_time()}] setting update tip pos={self.tip_pos_update_in_t[t, :]}, angle={self.tip_angle_update_in_t[t]}')
            prev_total_angle = 0.0
            self.restart = True
            self.last_restart_reason = "origin_cut"

        elif correct_for_coil and cond_coil:
            print(f'[{_local_log_time()}] coiled up too much at t={t}')
            self.origin_cut_restart_count = 0
            self.coil_count += 1

            # self.tip_pos_update_in_t[t, :] = self.tip_pos_in_t[t, :]
            # self.tip_angle_update_in_t[t] = self.tip_angle_in_t[t]
            # self.total_angle_update_in_t[t] = 0.0
            # print(f'setting update tip pos={self.tip_pos_update_in_t[t, :]}, angle={self.tip_angle_update_in_t[t]}')
            side_sign = np.sign(delta_angle)
            self._restart_flat_with_y_bias(t, Strctr, side_sign=side_sign)
            print(f'[{_local_log_time()}] setting update tip pos={self.tip_pos_update_in_t[t, :]}, angle={self.tip_angle_update_in_t[t]}')
            prev_total_angle = 0.0
            self.restart = True
            self.last_restart_reason = "coil"

        elif correct_for_update_force and cond_tip_force:
            print(f'[{_local_log_time()}] update forces too big at t={t}')
            self.coil_count = 0
            self.origin_cut_restart_count = 0

            rand_update_tip_pos, rand_update_tip_angle = self._random_update_tip(Strctr)
            self.tip_pos_update_in_t[t, :] = rand_update_tip_pos
            self.tip_angle_update_in_t[t] = rand_update_tip_angle
            self.total_angle_update_in_t[t] = 0.0
            print(f'[{_local_log_time()}] setting update tip pos={self.tip_pos_update_in_t[t, :]}, angle={self.tip_angle_update_in_t[t]}')
            prev_total_angle = 0.0
            self.restart = True
            self.last_restart_reason = "forces"

        # invert sign of tip change if training fails
        cond_extremal_buckle = (np.array_equal(State_meas.buckle_arr, np.array([[1], [1], [1], [1]]))
                                or
                                np.array_equal(State_meas.buckle_arr, np.array([[-1], [-1], [-1], [-1]])))
        if (cond_cut_origin or cond_coil) and (cond_extremal_buckle):
            print(f'[{_local_log_time()}] conditions for inverting delta tip inside run met, inverting delta tip')
            self.invert_delta_tip = not self.invert_delta_tip
        # if self.coil_count > 1 or self.origin_cut_restart_count > 1:
        #     print(f'inverting tip sign at time t={t}')
        #     self.invert_delta_tip = True

        if not self.supress_prints:
            delta_tip_after_corr = self.tip_pos_update_in_t[t, :] - prev_tip_update_pos
            delta_angle_after_corr = self.tip_angle_update_in_t[t] - prev_tip_update_angle
            print(f'delta_tip after correcting coil and cut origin {delta_tip_after_corr}')
            print(f'delta_angle after correcting coil and cut origin {delta_angle_after_corr}')

        # ------ update total angle -------
        self.total_angle_update_in_t[t] = helpers_builders._get_total_angle(self.tip_pos_update_in_t[t, :],
                                                                            prev_total_angle, Strctr.L)
        if not self.supress_prints:
            print(f'total angle end of calc_update {self.total_angle_update_in_t[t]}')

    # ---------------------------------------------------------------
    # Helpers (numpy)
    # ---------------------------------------------------------------
    def _random_update_tip(self, Strctr: "StructureClass") -> tuple[np.ndarray, float]:
        """
        Sample a random tip position uniformly inside a disk of given radius,
        and a random tip angle.

        Returns
        -------
        tip_pos : np.ndarray, shape (2,)
        tip_angle : float
        """
        tip_angle = np.pi / 2 * self.rng_tip.random()

        # uniform in disk
        R_eff = helpers_builders.effective_radius(self.R_free, Strctr.L, total_angle=0, tip_angle=tip_angle)
        r_min = 0.75 * R_eff
        r = r_min + (R_eff - r_min) * np.sqrt(self.rng_tip.random())
        phi = np.pi / 4 * self.rng_tip.random()
        tip_pos = np.array([r * np.cos(phi), r * np.sin(phi)], dtype=float)

        return tip_pos, tip_angle

    def _restart_flat_with_y_bias(self, t: int, Strctr: "StructureClass", side_sign: float) -> None:
        """
        Restart from flat, but bias the tip slightly above/below the x-axis.

        Repeated origin cuts increase the vertical bias magnitude.
        """
        mag = (self.origin_restart_base_frac +
               max(0, self.origin_cut_restart_count - 1) * self.origin_restart_step_frac) * Strctr.L

        y_restart = float(side_sign) * mag
        x_restart = float(Strctr.edges * Strctr.L - mag)
        tip_restart = np.array([x_restart, y_restart], dtype=np.float32)
        total_angle_restart = helpers_builders._get_total_angle(tip_restart, prev_total_angle=0.0, L=Strctr.L)

        self.tip_pos_update_in_t[t, :] = tip_restart
        self.tip_angle_update_in_t[t] = total_angle_restart
        self.total_angle_update_in_t[t] = total_angle_restart
        self.restart = True

        if not self.supress_prints:
            print(f"[{_local_log_time()}] cut origin for the {self.origin_cut_restart_count} time")

    def _get_loss_dispatch(self):
        return {"force": self._loss_force,
                "position": self._loss_tip_pos,
                }

    def _loss_force(self, Variabs, t, Fx=None, Fy=None, pos=None, pos_des=None, Strctr=None):
        loss = array([self.desired_Fx_in_t[t] - Fx,
                      self.desired_Fy_in_t[t] - Fy], dtype=np.float32)
        return loss / Variabs.norm_force

    def _loss_tip_pos(self, Variabs, t, Fx=None, Fy=None, pos=None, pos_des=None, Strctr=None):
        theta_meas = self._tip_angle_from_hinges(pos, Strctr)
        theta_des = self._tip_angle_from_hinges(pos_des, Strctr)
        loss_pos = (pos_des[-1] - pos[-1]) / Variabs.norm_pos
        loss_angle = (theta_des - theta_meas) / Variabs.norm_angle
        return np.append(loss_pos, loss_angle)

    @staticmethod
    def _tip_angle_from_hinges(pos_arr: NDArray, Strctr: Optional["StructureClass"] = None) -> float:
        """Return accumulated tip orientation from hinge angles when structure is available."""
        if Strctr is None:
            return float(helpers_builders._get_tip_angle(pos_arr))
        return float(np.sum(Strctr.all_hinge_angles(pos_arr)))

    def _get_delta_dispatch(self):
        """
        Map update_scheme -> function that computes (delta_tip_x, delta_tip_y, delta_angle).
        Each function must return 3 scalars in *your current convention*.

        Returns:
        --------
        function that calculates tip update values inside self.calc_update
        """
        return {
            "one_to_one": self._delta_one_to_one,
            "loss_diff": self._delta_loss_diff,
            "loss_x_trend": self._delta_loss_x_trend,
            "pos": self._delta_pos
            # "radial_one_to_one": self._delta_radial_one_to_one,
            # "lossx_concavity": self._lossx_concavity
            # "BEASTAL": self._delta_BEASTAL,
            # "radial_BEASTAL": self._delta_radial_BEASTAL,
            # "BEASTAL_no_pinv": self._delta_BEASTAL_no_pinv,
            # "radial_halfway_BEASTAL": self._delta_radial_halfway_BEASTAL,
        }

    def _delta_one_to_one(self, t, Strctr, Variabs, State_meas):
        """
        change tip directly from loss, no pseudo inverse, calculations in cartesian coordinates
        dx = +alpha*loss_x*sign(y)
        dy = -alpha*loss_x*sign(x)
        dtheta = -alpha*loss_y

        Parameters:
        -----------
        t : int, current training time step

        Returns:
        --------
        3 floats of change in tip position during update
        """
        sgnx = np.sign(self.tip_pos_update_in_t[t-1, 0])
        # sgny = np.sign(self.tip_pos_update_in_t[t - 1, 0])
        sgny = np.sign(self.tip_pos_update_in_t[t-1, 1])
        sgntheta_meas = np.sign(self.tip_angle_in_t[t])
        sgnlossx = np.sign(self.loss[0])
        sgnlossy = np.sign(self.loss[1])
        if sgnx == 0.0:
            sgnx = 1
        if sgny == 0.0:
            sgny = 1
        # delta_tip_x = + self.alpha * self.loss[0] * Strctr.hinges * Variabs.norm_pos * sgnx
        # delta_tip_y = - self.alpha * self.loss[0] * Strctr.hinges * Variabs.norm_pos * sgnx
        # delta_tip_x = - self.alpha * self.loss[0] * (-sgny) * Strctr.hinges * Variabs.norm_pos
        # delta_tip_y = - self.alpha * self.loss[0] * (+sgnx) * Strctr.hinges * Variabs.norm_pos
        delta_tip_x = - self.alpha * self.loss[0] * (-sgny) * Variabs.norm_pos  # up to Mar17
        delta_tip_y = - self.alpha * self.loss[0] * (+sgnx) * Variabs.norm_pos  # up to Mar17
        # delta_tip_x = - self.alpha * self.loss[0] * sgnlossy * (-sgny) * Variabs.norm_pos  # up to Mar17
        # delta_tip_y = - self.alpha * self.loss[0] * sgnlossy * (+sgnx) * Variabs.norm_pos  # up to Mar17
        # delta_tip_x = - self.alpha * self.loss[0] * (-sgntheta_meas) * (-sgny) * Variabs.norm_pos  # Mar18 improve_training
        # delta_tip_y = - self.alpha * self.loss[0] * (-sgntheta_meas) * (+sgnx) * Variabs.norm_pos  # Mar18 improve_training
        # delta_tip_x = - self.alpha * self.loss[0] * (-sgnlossx * sgnlossy) * (-sgny) * Variabs.norm_pos
        # delta_tip_y = - self.alpha * self.loss[0] * (-sgnlossx * sgnlossy) * (+sgnx) * Variabs.norm_pos  # up to Mar17
        # delta_angle = - self.alpha * self.loss[1] * Variabs.norm_angle * np.pi
        delta_angle = - self.alpha * self.loss[1] * Variabs.norm_angle  # up to Mar17
        # delta_angle = - self.alpha * self.loss[1] * (-sgnlossx) * Variabs.norm_angle  #
        return delta_tip_x, delta_tip_y, delta_angle

    def _delta_loss_diff(self, t, Strctr, Variabs, State_meas):
        sgnx = np.sign(self.tip_pos_update_in_t[t-1, 0])
        sgny = np.sign(self.tip_pos_update_in_t[t-1, 1])
        if sgnx == 0.0:
            sgnx = 1
        if sgny == 0.0:
            sgny = 1
        loss_diff = self.loss[0] - self.loss[1]
        loss_add = self.loss[0] + self.loss[1]
        delta_tip_x = - self.alpha * loss_diff * (-sgny) * Variabs.norm_pos  # Mar23
        delta_tip_y = - self.alpha * loss_diff * (+sgnx) * Variabs.norm_pos  # Mar23
        # delta_tip_x = - self.alpha * loss_diff * sgnLossx * (-sgnLossy) * (-sgny) * Variabs.norm_pos  # Mar24
        # delta_tip_y = - self.alpha * loss_diff * sgnLossx * (-sgnLossy) * (+sgnx) * Variabs.norm_pos  # Mar24

        delta_angle = - self.alpha * loss_add * Variabs.norm_angle  # Mar23
        # delta_angle = - self.alpha * loss_add * sgnLossx * (-sgnLossy) * Variabs.norm_angle  # Mar24
        return delta_tip_x, delta_tip_y, delta_angle

    def _delta_loss_x_trend(self, t, Strctr, Variabs, State_meas):
        Lx = self.loss[0]
        Lx_trend = self.loss_x_trend
        print('trend=', Lx_trend)
        Ly = self.loss[1]
        tradeoff_pos_theta = 1/4
        sgnx = np.sign(self.tip_pos_update_in_t[t-1, 0])
        sgny = np.sign(self.tip_pos_update_in_t[t-1, 1])
        if sgnx == 0.0:
            sgnx = 1
        if sgny == 0.0:
            sgny = 1
        loss_add = Lx + Ly
        delta_tip_x = - self.alpha * Lx_trend * tradeoff_pos_theta * (-sgny) * Variabs.norm_pos  # Mar23
        delta_tip_y = - self.alpha * Lx_trend * tradeoff_pos_theta * (+sgnx) * Variabs.norm_pos  # Mar23
        delta_angle = - self.alpha * loss_add * Variabs.norm_angle  # Mar23
        return delta_tip_x, delta_tip_y, delta_angle

    def _delta_pos(self, t, Strctr, Variabs, State_meas):

        x_rel = self.tip_pos_update_in_t[t-1, 0] - Strctr.L/2
        sgnx_update = np.sign(x_rel)
        sgny_update = np.sign(self.tip_pos_update_in_t[t-1, 1])
        sgnx_meas = np.sign(State_meas.pos_arr[-1][0])
        sgny_meas = np.sign(State_meas.pos_arr[-1][1])

        if sgnx_update == 0.0:
            sgnx_update = 1
        if sgny_update == 0.0:
            sgny_update = 1
        if sgnx_meas == 0.0:
            sgnx_meas = 1
        if sgny_meas == 0.0:
            sgny_meas = 1

        if self.pos_delta_mode == "signed":
            delta_tip_x = - self.alpha * (-self.loss[0]) * (-sgny_update) * Variabs.norm_pos  # July8
            delta_tip_y = - self.alpha * (-self.loss[1]) * (+sgnx_update) * Variabs.norm_pos  # July8
        elif self.pos_delta_mode == "signed_double":
            delta_tip_x = - self.alpha * (-self.loss[0]) * (-sgny_update) * (-sgny_meas) * Variabs.norm_pos  # Mar23
            delta_tip_y = - self.alpha * (-self.loss[1]) * (+sgnx_update) * (+sgnx_meas) * Variabs.norm_pos  # Mar23
        elif self.pos_delta_mode == "direct":
            delta_tip_x = - self.alpha * (-self.loss[0]) * Variabs.norm_pos  # May3 for rotation matrix
            delta_tip_y = - self.alpha * (-self.loss[1]) * Variabs.norm_pos  # May3 for rotation matrix
        else:
            raise ValueError(f"Unknown pos_delta_mode='{self.pos_delta_mode}'")
        delta_angle = - self.alpha * (-self.loss[2]) * Variabs.norm_angle  # Mar23

        # # angle for rotation of delta update
        # # Initial total angle. For free_tip flat this is basically 0, but compute it for generality.
        # state_meas_tip = State_meas.pos_arr_in_t[-2:, :, t][0]
        # theta0 = helpers_builders._get_total_angle(state_meas_tip, prev_total_angle=0.0, L=Strctr.L)
        # # theta0 = helpers_builders._get_tip_angle(State_meas.pos_arr_in_t[:, :, t])

        # # Use the previous accepted update angle, since the new one is not known yet.
        # if t <= 1:
        #     theta = theta0
        # else:
        #     theta = float(self.total_angle_update_in_t[t-1])
        #     # theta = float(self.tip_angle_update_in_t[t-1])

        # delta_tip_xy_rot = helpers_builders._rot2(theta - theta0) @ np.array([delta_tip_x, delta_tip_y])
        # return float(delta_tip_xy_rot[0]), float(delta_tip_xy_rot[1]), float(delta_angle)
        return float(delta_tip_x), float(delta_tip_y), float(delta_angle)

    # def _lossx_concavity(self, t, Strctr, Variabs, State_meas, State_des):
    #     sgnx = np.sign(self.tip_pos_update_in_t[t-1, 0])
    #     sgny = np.sign(self.tip_pos_update_in_t[t-1, 1])
    #     if sgnx == 0.0:
    #         sgnx = 1
    #     if sgny == 0.0:
    #         sgny = 1
    #     sgnLossy = np.sign(self.loss[1])
    #     delta_tip_x = - self.alpha * (-self.loss[1]) * (-sgny) * Variabs.norm_pos  # Mar25
    #     delta_tip_y = - self.alpha * (-self.loss[1]) * (+sgnx) * Variabs.norm_pos  # Mar25

    #     delta_angle = - self.alpha * (-self.concavity) * Variabs.norm_angle  # Mar24
    #     return delta_tip_x, delta_tip_y, delta_angle

    # def _delta_radial_one_to_one(self, t, Strctr, Variabs, State_meas, State_des):
    #     """
    #     change tip directly from loss, no pseudo inverse, calculations in polar coordinates
    #     dx = -alpha*loss_Theta*y!
    #     dy = -alpha*loss_Theta*(-x!)
    #     dtheta = -alpha*loss_tip

    #     Parameters:
    #     ------------
    #     t                 : current training time step
    #     current_tip_pos   : np.array(float) (2,), during measurement, i.e. Sprvsr.tip_pos_in_t[t]
    #     current_tip_angle : float, during measurement, i.e. Sprvsr.tip_angle_in_t[t]

    #     Returns:
    #     --------
    #     3 floats of change in tip position during update
    #     """
    #     if t == 1:
    #         prev_total_angle = helpers_builders._get_total_angle(self.tip_pos_in_t[t], 0.0, Strctr.L)
    #         tip_update = self.tip_pos_in_t[t]
    #     else:
    #         prev_total_angle = self.total_angle_update_in_t[t - 1]
    #         tip_update = self.tip_pos_update_in_t[t - 1, :]

    #     # loss in direction perpindicular to the total chain angle, measured from end of 2nd link
    #     loss_total_angle = helpers_builders._get_scalar_in_orthogonal_dir(self.loss, prev_total_angle)
    #     # loss in direction perp. to just the tip angle
    #     loss_tip = helpers_builders._get_scalar_in_orthogonal_dir(self.loss, self.tip_angle_in_t[t])

    #     delta_tip_x = (- self.alpha * loss_total_angle) * tip_update[1]
    #     delta_tip_y = (- self.alpha * loss_total_angle) * -tip_update[0]
    #     delta_angle = - self.alpha * loss_tip * Variabs.norm_angle * 2
    #     return delta_tip_x, delta_tip_y, delta_angle

    # def _delta_BEASTAL(self, t, Strctr, Variabs, State, current_tip_pos, current_tip_angle):
    #     inputs_normalized = array([
    #         current_tip_pos[0] / Variabs.norm_pos,
    #         current_tip_pos[1] / Variabs.norm_pos,
    #         current_tip_angle / Variabs.norm_angle
    #     ], dtype=np.float32)

    #     outputs_normalized = array([
    #         State.Fx / Variabs.norm_force,
    #         State.Fy / Variabs.norm_force
    #     ], dtype=np.float32)

    #     grad_loss_vec = learning_funcs.grad_loss_FC(
    #         Strctr.NE, inputs_normalized, outputs_normalized,
    #         Strctr.DM, Strctr.output_nodes_arr, self.loss
    #     )
    #     update_vec = - self.alpha * np.matmul(Strctr.DM_dagger, grad_loss_vec)

    #     delta_tip_x = update_vec[0] * Variabs.norm_pos
    #     delta_tip_y = update_vec[1] * Variabs.norm_pos
    #     delta_angle = - update_vec[2] * Variabs.norm_angle
    #     return delta_tip_x, delta_tip_y, delta_angle

    # def _delta_radial_BEASTAL(self, t, Strctr, Variabs, State, current_tip_pos, current_tip_angle):
    #     if t == 1:
    #         prev_total_angle = 0.0
    #         tip_update = current_tip_pos
    #     else:
    #         prev_total_angle = self.total_angle_update_in_t[t - 1]
    #         tip_update = self.tip_pos_update_in_t[t - 1, :]

    #     total_angle_meas = helpers_builders._get_total_angle(current_tip_pos, prev_total_angle, Strctr.L)

    #     loss_total_angle = helpers_builders._get_scalar_in_orthogonal_dir(self.loss, total_angle_meas)
    #     F_total_angle = helpers_builders._get_scalar_in_orthogonal_dir(array([State.Fx, State.Fy]), total_angle_meas)

    #     loss_tip = helpers_builders._get_scalar_in_orthogonal_dir(self.loss, current_tip_angle)
    #     F_tip_angle = helpers_builders._get_scalar_in_orthogonal_dir(array([State.Fx, State.Fy]), current_tip_angle)

    #     inputs_normalized = array([
    #         total_angle_meas / Variabs.norm_angle,
    #         current_tip_angle / Variabs.norm_angle
    #     ], dtype=np.float32)

    #     outputs_normalized = array([
    #         F_total_angle / Variabs.norm_force,
    #         F_tip_angle / Variabs.norm_force
    #     ], dtype=np.float32)

    #     d_total_angle = - self.alpha * 1/8 * (
    #         loss_total_angle * (3*outputs_normalized[0] - outputs_normalized[1] - 2*inputs_normalized[0]) +
    #         loss_tip       * (3*outputs_normalized[0] - outputs_normalized[1] - 2*inputs_normalized[1])
    #     )
    #     d_tip_angle = - self.alpha * 1/8 * (
    #         loss_total_angle * (3*outputs_normalized[1] - outputs_normalized[0] - 2*inputs_normalized[0]) +
    #         loss_tip       * (3*outputs_normalized[1] - outputs_normalized[0] - 2*inputs_normalized[1])
    #     )

    #     loss_thetas = array([loss_total_angle, loss_tip])

    #     grad_loss_vec = learning_funcs.grad_loss_FC(
    #         Strctr.NE, inputs_normalized, outputs_normalized,
    #         Strctr.DM, Strctr.output_nodes_arr, loss_thetas
    #     )

    #     # kept exactly as in your code (even if grad_loss_vec is unused below)
    #     predicted_grad_loss_1 = (outputs_normalized[0] - inputs_normalized[0]) * loss_total_angle
    #     predicted_grad_loss_2 = (outputs_normalized[1] - inputs_normalized[0]) * loss_tip

    #     update_vec = array([-d_total_angle, -d_tip_angle]) * np.sign(total_angle_meas)

    #     delta_tip_x = update_vec[0] * tip_update[1]
    #     delta_tip_y = update_vec[0] * -tip_update[0]
    #     delta_angle = update_vec[1] * Variabs.norm_angle
    #     return delta_tip_x, delta_tip_y, delta_angle

    # def _delta_BEASTAL_no_pinv(self, t, Strctr, Variabs, State, current_tip_pos, current_tip_angle):
    #     delta_tip_x = + self.alpha * self.loss[0] / Variabs.norm_force * Strctr.hinges * Strctr.L * (
    #         current_tip_pos[0] - Strctr.hinges * Strctr.L
    #     )
    #     delta_tip_y = - self.alpha * self.loss[1] / Variabs.norm_force * Strctr.hinges * Strctr.L * current_tip_pos[1]
    #     if self.loss.size == 3:
    #         delta_angle = + self.alpha * self.loss[2] / Variabs.norm_torque * np.pi/64 * current_tip_angle
    #     else:
    #         delta_angle = 0.0
    #     return delta_tip_x, delta_tip_y, delta_angle

    # def _delta_radial_halfway_BEASTAL(self, t, Strctr, Variabs, State, current_tip_pos, current_tip_angle):
    #     if t == 1:
    #         prev_total_angle = 0.0
    #         tip_update = current_tip_pos
    #     else:
    #         prev_total_angle = self.total_angle_update_in_t[t - 1]
    #         tip_update = self.tip_pos_update_in_t[t - 1, :]

    #     total_angle = helpers_builders._get_total_angle(current_tip_pos, 0.0, Strctr.L)
    #     print(f'total_angle_for_loss={total_angle}')
    #     loss_total_angle = -self.loss[0] * np.sin(total_angle) + self.loss[1] * np.cos(total_angle)
    #     print(f'loss_total_angle={loss_total_angle:.2f}')
    #     loss_tip = -self.loss[0] * np.sin(current_tip_angle) + self.loss[1] * np.cos(current_tip_angle)
    #     print(f'tip_angle_for_loss={current_tip_angle}')
    #     print(f'loss_tip={loss_tip:.2f}')
    #     F_total_angle = helpers_builders._get_scalar_in_orthogonal_dir(array([State.Fx, State.Fy]), total_angle)
    #     print(f'F_total_angle={F_total_angle}')
    #     F_tip_angle = helpers_builders._get_scalar_in_orthogonal_dir(array([State.Fx, State.Fy]), current_tip_angle)
    #     print(f'F_tip_angle={F_tip_angle}')

    #     delta_tip_x = (self.alpha * loss_total_angle) * tip_update[1] * F_total_angle / Variabs.norm_force
    #     delta_tip_y = (self.alpha * loss_total_angle) * -tip_update[0] * F_total_angle / Variabs.norm_force
    #     delta_angle = self.alpha * loss_tip * Variabs.norm_angle * 2 * F_tip_angle / Variabs.norm_force
    #     return delta_tip_x, delta_tip_y, delta_angle
