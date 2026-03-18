from __future__ import annotations

import numpy as np
import copy
import jax
import jax.numpy as jnp
import equinox as eqx
from jax import vmap

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
    loss_in_t          : ndarray (T, 2), loss (x and y) for every measurement during training.
    loss_MSE_in_t      : ndarray (T,), Mean Squared Error of the x,y loss.
    tip_pos_update_in_t: ndarray (T, 2), tip position in the Update modality, for every training step.
    tip_angle_update_in_t : ndarray (T, ), tip angle in the Update modality, for every training step.
    total_angle_update_in_t : ndarray (T, ), angle between tip and end of first link, in Update modality, for every training step
    imposed_mask       : # (2*nodes,), boolean of whether a node (ends of edges/facets) is imposed or not
                         True only at two final nodes, if control_tip==True
    loss               : (2,), instantaneous loss
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
    control_tip: bool = eqx.field(static=True)
    control_first_edge: bool = eqx.field(static=True)
    normalize_step: bool = eqx.field(static=True)
    R_free: float = eqx.field(static=True)
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
    tip_angle_in_t: Optional[NDArray[np.float32]] = eqx.field(default=None, init=False, static=True)    # (T,)

    # --- running logs / losses ---
    loss_in_t: NDArray[np.float32] = eqx.field(init=False, static=True)            # (T, 2)
    loss_MSE_in_t: NDArray[np.float32] = eqx.field(init=False, static=True)        # (T,)
    tip_pos_update_in_t: NDArray[np.float32] = eqx.field(init=False, static=True)  # (T, 2)
    tip_angle_update_in_t: Optional[NDArray[np.float32]] = eqx.field(default=None, init=False,
                                                                     static=True)  # (T,)
    total_angle_update_in_t: Optional[NDArray[np.float32]] = eqx.field(default=None, init=False,
                                                                       static=True)  # (T,)

    # ------ for equilibrium calculation, jax arrays ------
    imposed_mask: jax.ndarray[bool] = eqx.field(static=True)                       # (2*nodes,)

    # --- scratch (most recent loss vector) ---
    loss: NDArray[np.float32] = eqx.field(init=False, static=True)                 # (2,)

    def __init__(self, Strctr, CFG, supress_prints: bool = True) -> None:
        self.T = int(CFG.Train.T)  # total training-set size (& algorithm time, not to confuse with time to equilib state)
        self.alpha = float(CFG.Train.alpha)
        self.update_scheme = str(CFG.Train.update_scheme)
        self.control_tip = bool(CFG.Train.control_tip)
        self.control_first_edge = bool(CFG.Train.control_first_edge)  # if true, fix nodes (0, 1), else fix only node (0)

        # for equilibrium
        self.imposed_mask = self._build_imposed_mask(Strctr, self.control_tip)

        # Desired/targets
        if CFG.Train.desired_buckle_type == 'random':  # uniformly distributed values of +1 and -1
            key = jax.random.PRNGKey(CFG.Train.desired_buckle_rand_key)   # seed
            desired_buckle = jax.random.randint(key, (Strctr.hinges, Strctr.shims), minval=-1, maxval=2)  # +1, 0 or -1
            desired_buckle = desired_buckle.at[desired_buckle == 0].set(-1)  # replace 0 w/ -1
        elif CFG.Train.desired_buckle_type == 'opposite':  # opposite than initial buckle, requires creating the initial buckle
            desired_buckle = - helpers_builders._initiate_buckle(Strctr.hinges, Strctr.shims,
                                                                 buckle_pattern=CFG.Train.init_buckle_pattern, numpify=True)
        elif CFG.Train.desired_buckle_type == 'straight':  # same as initial buckle, requires creating the initial buckle
            desired_buckle = helpers_builders._initiate_buckle(Strctr.hinges, Strctr.shims,
                                                               buckle_pattern=CFG.Train.init_buckle_pattern, numpify=True)
        elif CFG.Train.desired_buckle_type == 'specified':
            desired_buckle = helpers_builders._initiate_buckle(Strctr.hinges, Strctr.shims,
                                                               buckle_pattern=CFG.Train.desired_buckle_pattern, numpify=True)
        self.desired_buckle_arr = np.asarray(desired_buckle, dtype=np.int32)
        self.desired_pos_in_t = zeros((Strctr.nodes, 2, self.T), dtype=np.float32)
        self.desired_Fx_in_t = zeros((self.T), dtype=np.float32)
        self.desired_Fy_in_t = zeros((self.T), dtype=np.float32)

        # Dataset (commands)
        self.tip_pos_in_t = zeros((self.T, 2), dtype=np.float32)
        self.tip_angle_in_t = zeros((self.T,), dtype=np.float32)

        # Logs / updates
        loss_size = 2
        self.loss_in_t = zeros((self.T, loss_size), dtype=np.float32)
        self.loss_MSE_in_t = zeros((self.T,), dtype=np.float32)

        # Last loss vector (shape matches control mode)
        self.loss = zeros(loss_size, dtype=np.float32)

        self.tip_pos_update_in_t = zeros((self.T, 2), dtype=np.float32)
        self.tip_angle_update_in_t = zeros((self.T,), dtype=np.float32)
        self.total_angle_update_in_t = zeros((self.T,), dtype=np.float32)

        self.normalize_step = bool(CFG.Train.normalize_step)  # whether to normalize the training step in [x, y, theta] space

        self.R_free = (Strctr.edges - 2*0.98)*Strctr.L  # maximal radius the chain could have, up to some margin

        # for output files
        self.convert_pos = CFG.Train.convert_pos
        self.convert_angle = CFG.Train.convert_angle
        self.convert_F = CFG.Train.convert_F

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
        according to sampling strategy. These trajectories are used in measurement, update, or stress–strain protocols.

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

        # tip positions and angles for specified tip dataset
        if sampling == 'uniform':
            rng = np.random.default_rng(CFG.Train.rand_key_dataset)
            low = array([(Strctr.edges - 0.5) * Strctr.L, -Strctr.L * 1 / 3, -np.pi / 5])  # lowest allowed value
            high = array([(Strctr.edges - 0.01) * Strctr.L, Strctr.L * 1 / 3, np.pi / 5])  # highest allowed value
            samples = rng.uniform(low, high, size=(self.T, 3)).astype(np.float32)  # (T, 3) sample size
            self.tip_pos_in_t = samples[:, :2]  # (T, 2)
            self.tip_angle_in_t = samples[:, 2]  # (T,)

            # correct for too big stretch during measurement
            # ------ clamp overstretched dataset samples ------
            for t in range(self.T):
                self.tip_pos_in_t[t, :] = helpers_builders._correct_big_stretch_robot_style(tip_pos=self.tip_pos_in_t[t, :], 
                                                                                            tip_angle=float(self.tip_angle_in_t[t]),
                                                                                            total_angle=0.0, R_free=self.R_free,
                                                                                            L=Strctr.L, margin=0.1,
                                                                                            supress_prints=self.supress_prints)
        elif sampling in {'flat', 'almost_flat', 'specified'}:
            end = float(Strctr.edges*Strctr.L)
            if sampling == 'flat':
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
        elif sampling == "tile":
            self.tip_pos_in_t[:] = np.tile(tip_pos, (self.T // len(tip_pos) + 1, 1))[:self.T]
            tip_angles_block = np.repeat(tip_angle, tip_pos.shape[0])
            self.tip_angle_in_t[:] = np.tile(tip_angles_block, self.T // len(tip_angles_block) + 1)[:self.T]
        else:
            raise ValueError(f"Incompatible sampling='{sampling}'")

    def set_desired(self, pos_arr: jax.Array, Fx: float, Fy: float, t: int) -> None:
        """Store ground-truth targets for step t.

        Parameters:
        -----------
        pos_arr : (T, 2*N) node positions in x and y [mm]
        Fx      : float force in global x direction
        Fy      : float force in global y direction
        t       : {0:self.T} current time step
        """
        self.desired_pos_in_t[:, :, t] = helpers_builders.jax2numpy(pos_arr)
        self.desired_Fx_in_t[t] = float(Fx)
        self.desired_Fy_in_t[t] = float(Fy)

    # ---------------------------------------------------------------
    # Calculations - loss and Update values
    # ---------------------------------------------------------------
    def calc_loss(self, Variabs: "VariablesClass", t: int, Fx: float, Fy: float) -> None:
        """Compute loss vector (Fx,Fy) at step t and log it.

        Parameters:
        -----------
        Variabs : VariablesClass, using: 
                  - norm_force: float typical force calculated in Variabs.init
        t       : {0:self.T} current time step
        Fx      : float force in global x direction
        Fy      : float force in global y direction

        Returns:
        --------
        loss     - float, F_hat-F in 2d
        loss_MSE - float, mean squared loss
        """
        self.loss = array([self.desired_Fx_in_t[t] - Fx, self.desired_Fy_in_t[t] - Fy], dtype=np.float32)

        # normalize loss
        self.loss = self.loss / Variabs.norm_force

        # put in loss vec
        self.loss_in_t[t, : self.loss.shape[0]] = self.loss

        # same for Mean Squared Error
        self.loss_MSE = np.sqrt(np.sum(self.loss**2))
        self.loss_MSE_in_t[t] = self.loss_MSE

    def calc_update_tip(self, t: int, Strctr: "StructureClass", Variabs: "VariablesClass",
                        current_tip_pos: Optional[np.ndarray] = None,
                        prev_tip_update_pos: Optional[np.ndarray] = None,
                        current_tip_angle: Optional[float] = None,
                        prev_tip_update_angle: Optional[float] = None,
                        correct_for_total_angle: Optional[bool] = False,
                        correct_for_coil: Optional[bool] = True,
                        correct_for_cut_origin: Optional[bool] = True) -> None:
        """Compute next tip position/angle commands from current loss and state (pure NumPy).

        Parameters:
        -----------
        t
        current_tip_pos       : ndarrat(float) (2,) during measurement, used only in radial_one_to_one update function
        prev_tip_update_pos   : ndarrat(float) (2,) previous update tip pos, for inserting new top into vectors in time
        current_tip_angle     : float, during measurement, used only in radial_one_to_one update function
        prev_tip_update_angle : float, previous update tip pos, for inserting new top into vectors in time
        correct_for_total_angle, correct_for_coil, correct_for_cut_origin : booleans, whether to correct tip pos due to:
                                                                            addition to simulation total angle,
                                                                            coiled tip (reset tip values from dataset)
                                                                            tip cuts origin (as above)

        Returns:
        --------
        updates in self:
        np.array(float), (2,) update_tip_pos_in_t
        float, tip_angle_update_in_t
        float, total_angle_update_in_t
        """
        # ------ delta tip and angle ------
        # through BEASTAL, one_to_one or radial_one_to_one
        dispatch = self._get_delta_dispatch()
        fn = dispatch.get(self.update_scheme, None)  # function to calculate delta tip and angle from update_scheme
        if fn is None:
            raise ValueError(f"Unknown update_scheme='{self.update_scheme}'")
        delta_tip_x, delta_tip_y, delta_angle = fn(t, Strctr, Variabs, current_tip_pos=current_tip_pos,
                                                   current_tip_angle=current_tip_angle)
        delta_tip = array([delta_tip_x, delta_tip_y])  # assemble into 3d array
        if not self.supress_prints:
            print(f'delta_tip before corr {delta_tip}')
            print(f'delta_angle before corr {delta_angle}')

        # ------ normalize step ------
        if self.normalize_step and np.linalg.norm(np.append(delta_tip, delta_angle)) > 10**(-12):  # normalize if non-zero update            
            # old version up to Feb22
            step_size = np.linalg.norm(np.append(delta_tip, delta_angle))
            # print(f'step_size={step_size}')
            tradeoff_pos_angle = 1/2
            delta_tip = copy.copy(delta_tip)/step_size*self.alpha
            delta_angle = copy.copy(delta_angle)/step_size*self.alpha * tradeoff_pos_angle

            # new version from Feb22
            # pos_step_size = np.linalg.norm(delta_tip)
            # angle_step_size = np.linalg.norm(delta_angle)
            # tradeoff_pos_angle = 1
            # delta_tip = copy.copy(delta_tip)/pos_step_size*self.alpha
            # delta_angle = copy.copy(delta_angle)*(angle_step_size/pos_step_size)*self.alpha * tradeoff_pos_angle

            if not self.supress_prints:
                print(f'normalized position to {delta_tip}')
                print(f'normalized angle to {float(delta_angle)}')

        # ------ insert into vectors in time ------
        # insert into tip_pos_update
        if prev_tip_update_pos is None:
            prev_tip_update_pos = self.tip_pos_update_in_t[t-1, :]
        if prev_tip_update_angle is None:
            prev_tip_update_angle = self.tip_angle_update_in_t[t-1]

        # Angle update (only if enabled)
        if prev_tip_update_angle is None and t > 0:
            prev_tip_update_angle = float(self.tip_angle_update_in_t[t - 1])
        elif prev_tip_update_angle is None:
            prev_tip_update_angle = current_tip_angle

        if not self.supress_prints:
            print(f'prev_tip_update_pos{prev_tip_update_pos}')
            print(f'prev_tip_update_angle{prev_tip_update_angle}')

        # add to tip position in time
        self.tip_pos_update_in_t[t, :] = prev_tip_update_pos + delta_tip
        self.tip_angle_update_in_t[t] = prev_tip_update_angle + float(delta_angle)

        # ------ correct for total angle ------
        # add change in tip angle to the total angle from the origin
        if correct_for_total_angle:
            if t == 1:
                prev_total_angle = 0.0
            else:
                prev_total_angle = self.total_angle_update_in_t[t-1]
            total_angle = helpers_builders._get_total_angle(self.tip_pos_update_in_t[t, :], prev_total_angle, Strctr.L)
            self.total_angle_update_in_t[t] = total_angle
            delta_total_angle = total_angle - prev_total_angle
            self.tip_angle_update_in_t[t] += delta_total_angle
            if not self.supress_prints:
                print(f'total angle {total_angle}')
                print(f'add delta tip angle {delta_total_angle} to correct for total angle ')

        # ------ correct for to big a stretch ------
        self.tip_pos_update_in_t[t, :] = helpers_builders._correct_big_stretch_robot_style(self.tip_pos_update_in_t[t],
                                                                                           self.tip_angle_update_in_t[t],
                                                                                           total_angle, self.R_free, Strctr.L,
                                                                                           margin=0.1, 
                                                                                           supress_prints=self.supress_prints)
        if not self.supress_prints:
            print(f'tip after correct big stretch={self.tip_pos_update_in_t[t, :]}')
        # before_tip = helpers_builders._get_before_tip(prev_tip_update_pos, prev_tip_update_angle, Strctr.L, xp=np)
        # R_eff = helpers_builders.effective_radius(self.R_free, Strctr.L, total_angle, prev_tip_update_angle, 
        #                                           supress_prints=self.supress_prints)
        # tip_new, before_new, clamped = helpers_builders.clamp_pos_same_delta(before_prev=before_tip,
        #                                                                      tip_angle_new=prev_tip_update_angle + delta_angle,
        #                                                                      tip_raw=prev_tip_update_pos + delta_tip,
        #                                                                      second_node=array([Strctr.L, 0.0]),
        #                                                                      R_lim=R_eff, L=Strctr.L)
        # self.tip_pos_update_in_t[t, :] = tip_new
        # if not self.supress_prints:
        #     print(f'tip clamped due to effective radius: {clamped}')
        #     print(f'tip after clamp to radius={tip_new}')

        # ------ correct for coil or cut origin ------
        # if origin is cut or tip coiled too much, restart Update tip position and angle to current training set sample
        cond_coil = helpers_builders.coil(self.tip_angle_update_in_t[t], revolutions=1.5)
        # cond_cut_origin = np.linalg.norm(self.tip_pos_update_in_t[t, :]) < Strctr.L

        before_tip_tminus1 = helpers_builders._get_before_tip(self.tip_pos_update_in_t[t-1, :],
                                                              self.tip_angle_update_in_t[t-1], Strctr.L, xp=np)
        before_tip_t = helpers_builders._get_before_tip(self.tip_pos_update_in_t[t, :], self.tip_angle_update_in_t[t],
                                                        Strctr.L, xp=np)
        cond_cut_origin = helpers_builders.swept_last_edge_crosses_first_edge(before_prev=before_tip_tminus1,
                                                                              tip_prev=self.tip_pos_in_t[t-1, :],
                                                                              before_new=before_tip_t,
                                                                              tip_new=self.tip_pos_update_in_t[t, :],
                                                                              L=Strctr.L, include_endpoints=False)

        if correct_for_coil and cond_coil:
            print('coiled up too much')
            self.tip_pos_update_in_t[t, :] = self.tip_pos_in_t[t, :]
            self.tip_angle_update_in_t[t] = self.tip_angle_in_t[t]
            print(f'setting update tip pos={self.tip_pos_update_in_t[t, :]}, angle={self.tip_angle_update_in_t[t]}')
            prev_total_angle = 0.0  # reset previous total angle for total angle calculation in time=t

        if correct_for_cut_origin and cond_cut_origin:
            print('origin is cut')
            # tip_safe, angle_safe, corrected = helpers_builders.avoid_first_edge_crossing_same_step(
            #     before_prev=before_tip_tminus1,
            #     tip_prev=self.tip_pos_update_in_t[t-1, :],
            #     angle_prev=self.tip_angle_update_in_t[t-1],
            #     before_raw=before_tip_t,
            #     tip_raw=self.tip_pos_update_in_t[t, :],
            #     angle_raw=self.tip_angle_update_in_t[t],
            #     L=Strctr.L,
            #     include_endpoints=False,
            #     safety=1e-3,
            # )
            # delta_tip_slide = helpers_builders.evade_first_edge_by_sliding(tip_prev=self.tip_pos_update_in_t[t-1, :],
            #                                                                delta_tip_raw=delta_tip,
            #                                                                L=Strctr.L)
            # self.tip_pos_update_in_t[t, :] = self.tip_pos_update_in_t[t-1, :] + delta_tip_slide
            self.tip_pos_update_in_t[t, :] = self.tip_pos_in_t[t, :]
            self.tip_angle_update_in_t[t] = self.tip_angle_in_t[t]
            print(f'setting update tip pos={self.tip_pos_update_in_t[t, :]}, angle={self.tip_angle_update_in_t[t]}')

        if not self.supress_prints:
            delta_tip_after_corr = self.tip_pos_update_in_t[t, :] - self.tip_pos_update_in_t[t-1, :]
            delta_angle_after_corr = self.tip_angle_update_in_t[t] - self.tip_angle_update_in_t[t-1]
            print(f'delta_tip after correcting coil and cut origin {delta_tip_after_corr}')
            print(f'delta_angle after correcting coil and cut origin {delta_angle_after_corr}')

        # ------ update total angle -------
        self.total_angle_update_in_t[t] = helpers_builders._get_total_angle(self.tip_pos_update_in_t[t, :], prev_total_angle,
                                                                            Strctr.L)
        if not self.supress_prints:
            print(f'total angle end of calc_update {self.total_angle_update_in_t[t]}')

    # ---------------------------------------------------------------
    # Helpers (numpy)
    # ---------------------------------------------------------------
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
            "radial_one_to_one": self._delta_radial_one_to_one,
            # "BEASTAL": self._delta_BEASTAL,
            # "radial_BEASTAL": self._delta_radial_BEASTAL,
            # "BEASTAL_no_pinv": self._delta_BEASTAL_no_pinv,
            # "radial_halfway_BEASTAL": self._delta_radial_halfway_BEASTAL,
        }

    def _delta_one_to_one(self, t, Strctr, Variabs, current_tip_pos, current_tip_angle):
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
        sgnx = np.sign(self.tip_pos_update_in_t[t - 1, 0])
        # sgny = np.sign(self.tip_pos_update_in_t[t - 1, 0])
        sgny = np.sign(self.tip_pos_update_in_t[t-1, 1])
        if sgnx == 0.0:
            sgnx = 1
        if sgny == 0.0:
            sgny = 1
        # delta_tip_x = + self.alpha * self.loss[0] * Strctr.hinges * Variabs.norm_pos * sgnx
        # delta_tip_y = - self.alpha * self.loss[0] * Strctr.hinges * Variabs.norm_pos * sgnx
        # delta_tip_x = - self.alpha * self.loss[0] * (-sgny) * Strctr.hinges * Variabs.norm_pos
        # delta_tip_y = - self.alpha * self.loss[0] * (+sgnx) * Strctr.hinges * Variabs.norm_pos
        delta_tip_x = - self.alpha * self.loss[0] * (-sgny) * Variabs.norm_pos
        delta_tip_y = - self.alpha * self.loss[0] * (+sgnx) * Variabs.norm_pos
        # delta_angle = - self.alpha * self.loss[1] * Variabs.norm_angle * np.pi
        delta_angle = - self.alpha * self.loss[1] * Variabs.norm_angle
        return delta_tip_x, delta_tip_y, delta_angle

    def _delta_radial_one_to_one(self, t, Strctr, Variabs, current_tip_pos, current_tip_angle):
        """
        change tip directly from loss, no pseudo inverse, calculations in polar coordinates
        dx = -alpha*loss_Theta*y!
        dy = -alpha*loss_Theta*(-x!)
        dtheta = -alpha*loss_tip

        Parameters:
        ------------
        t                 : current training time step
        current_tip_pos   : np.array(float) (2,), during measurement, i.e. Sprvsr.tip_pos_in_t[t]
        current_tip_angle : float, during measurement, i.e. Sprvsr.tip_angle_in_t[t]

        Returns:
        --------
        3 floats of change in tip position during update
        """
        if t == 1:
            prev_total_angle = helpers_builders._get_total_angle(current_tip_pos, 0.0, Strctr.L)
            tip_update = current_tip_pos
        else:
            prev_total_angle = self.total_angle_update_in_t[t - 1]
            tip_update = self.tip_pos_update_in_t[t - 1, :]

        # loss in direction perpindicular to the total chain angle, measured from end of 2nd link
        loss_total_angle = helpers_builders._get_scalar_in_orthogonal_dir(self.loss, prev_total_angle)
        # loss in direction perp. to just the tip angle
        loss_tip = helpers_builders._get_scalar_in_orthogonal_dir(self.loss, current_tip_angle)

        delta_tip_x = (- self.alpha * loss_total_angle) * tip_update[1]
        delta_tip_y = (- self.alpha * loss_total_angle) * -tip_update[0]
        delta_angle = - self.alpha * loss_tip * Variabs.norm_angle * 2
        return delta_tip_x, delta_tip_y, delta_angle

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
