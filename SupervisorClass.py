from __future__ import annotations

import numpy as np
np.set_printoptions(precision=4, suppress=True)

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


# ===================================================
# Class - Supervisor Variables - training set, losses, etc.
# ===================================================


class SupervisorClass:
    """
    Variables that are by the external supervisor in the experiment

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

    def __init__(self, Strctr, CFG) -> None:
        self.T = int(CFG.Train.T)  # total training-set size (& algorithm time, not to confuse with time to equilib state)
        self.alpha = float(CFG.Train.alpha)
        self.update_scheme = str(CFG.Train.update_scheme)
        self.control_tip = bool(CFG.Train.control_tip)
        self.control_first_edge = bool(CFG.Train.control_first_edge)  # if true, fix nodes (0, 1), else fix only node (0)

        # for equilibrium
        self.imposed_mask = self._build_imposed_mask(Strctr, self.control_tip)

        # Desired/targets
        if CFG.Train.desired_buckle_type == 'random':  # uniformly distributed values of +1 and -1
            key = jax.random.PRNGKey(CFG.Traindesired_buckle_rand_key)   # seed
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
        self.desired_pos_in_t = np.zeros((Strctr.nodes, 2, self.T), dtype=np.float32)
        self.desired_Fx_in_t = np.zeros((self.T), dtype=np.float32)
        self.desired_Fy_in_t = np.zeros((self.T), dtype=np.float32)

        # Dataset (commands)
        self.tip_pos_in_t = np.zeros((self.T, 2), dtype=np.float32)
        self.tip_angle_in_t = np.zeros((self.T,), dtype=np.float32)

        # Logs / updates
        loss_size = 2
        self.loss_in_t = np.zeros((self.T, loss_size), dtype=np.float32)
        self.loss_MSE_in_t = np.zeros((self.T,), dtype=np.float32)

        # Last loss vector (shape matches control mode)
        self.loss = np.zeros(loss_size, dtype=np.float32)

        self.tip_pos_update_in_t = np.zeros((self.T, 2), dtype=np.float32)
        self.tip_angle_update_in_t = np.zeros((self.T,), dtype=np.float32)
        self.total_angle_update_in_t = np.zeros((self.T,), dtype=np.float32)

        self.normalize_step = bool(CFG.Train.normalize_step)  # whether to normalize the training step in [x, y, theta] space

        self.R_free = (Strctr.edges - 2*0.9)*Strctr.L  # maximal radius the chain could have, up to some margin

        # for output files
        self.convert_pos = CFG.Train.convert_pos
        self.convert_angle = CFG.Train.convert_angle
        self.convert_F = CFG.Train.convert_F

    def _build_imposed_mask(self, Strctr: "StructureClass", control_tip: bool = True):
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

    def create_dataset(self, Strctr: "StructureClass", CFG, sampling: str, tip_pos: Optional[NDArray] = None,
                       tip_angle: Optional[float] = None, dist_noise: float = 0.0, angle_noise: float = 0.0) -> None:
        """
        Fill in the tip positions and angles during Measurement modality.

        inputs:
        sampling     : str, optional, Method for generating the command dataset. One of:
                     'uniform'       = random uniform vals for x, y, angle
                     'almost flat'   = flat piece, single measurement
                     'stress strain' = immitate stress strain where compression is in x axis, incremental
        """
        # save as variable
        self.dataset_sampling = sampling

        # tip positions and angles for specified tip dataset
        if sampling == 'uniform':
            np.random.seed(CFG.Train.rand_key_dataset)
            x_pos_in_t = np.random.uniform((Strctr.edges-1.5)*Strctr.L, (Strctr.edges-0.5)*Strctr.L, size=self.T)
            y_pos_in_t = np.random.uniform(-Strctr.L/2, Strctr.L/2, size=self.T)
            self.tip_pos_in_t = np.stack(((x_pos_in_t), (y_pos_in_t.T)), axis=1)
            self.tip_angle_in_t[:] = np.random.uniform(-np.pi / 5, np.pi / 5, size=self.T).astype(np.float32)
        elif sampling == 'flat':
            end = float(Strctr.edges)
            tip_pos = np.array([end, 0], dtype=np.float32)
            self.tip_pos_in_t[:] = np.tile(tip_pos, (self.T, 1))
            self.tip_angle_in_t[:] = 0.0
        elif sampling == 'almost flat':
            end = float(Strctr.edges*Strctr.L)
            tip_pos = np.array([end-0.1*Strctr.L,  0.0*Strctr.L], dtype=np.float32)  # flat arrangement
            # tip_pos = np.array([end-0.4*Strctr.L,  0.4*Strctr.L], dtype=np.float32)  # flat arrangement

            # tiny noise around each position (tune scale as you like)
            noise_scale = 0.0 * Strctr.L
            noise_pos = noise_scale * np.random.randn(self.T, 2).astype(np.float32)
            noise_pos[:, 0] = -np.abs(noise_pos[:, 0])
            self.tip_pos_in_t[:] = tip_pos + noise_pos

            # noise_angle = noise_scale * np.random.randn(self.T,).astype(np.float32)
            # noise_angle = np.pi/16
            noise_angle = 0
            self.tip_angle_in_t[:] = noise_angle
        elif sampling == 'specified':
            self.tip_pos_in_t[:] = np.tile(tip_pos, (self.T, 1))
            self.tip_angle_in_t[:] = tip_angle
        elif sampling == 'stress strain':
            start = 2*Strctr.L + CFG.Variabs.exp_start
            end = start - CFG.Variabs.distance
            tip_in = np.linspace(start, end, self.T // 2, endpoint=False)  # decreasing: start -> end
            tip_out = np.linspace(end, start, self.T - self.T // 2, endpoint=False)  # increasing: end -> start
            tip_arr = np.concatenate([tip_in, tip_out])  # shape (self.T,),  back-and-forth trajectory

            noisy_zeros_arr = np.zeros_like(tip_arr) + dist_noise  # shape (N,)
            self.tip_pos_in_t[:] = np.column_stack((tip_arr, noisy_zeros_arr))  # shape (N, 2)
            self.tip_angle_in_t[:] = angle_noise
        elif sampling == "tile":
            self.tip_pos_in_t[:] = np.tile(tip_pos, (self.T // len(tip_pos) + 1, 1))[:self.T]
            tip_angles_block = np.repeat(tip_angle, tip_pos.shape[0])
            self.tip_angle_in_t[:] = np.tile(tip_angles_block, self.T // len(tip_angles_block) + 1)[:self.T]
        else:
            raise ValueError(f"Incompatible sampling='{sampling}'")

    def set_desired(self, pos_arr: jax.Array, Fx: float, Fy: float, t: int) -> None:
        """Store ground-truth targets for step t."""
        self.desired_pos_in_t[:, :, t] = helpers_builders.jax2numpy(pos_arr)
        self.desired_Fx_in_t[t] = float(Fx)
        self.desired_Fy_in_t[t] = float(Fy)

    def calc_loss(self, Variabs: "VariablesClass", t: int, Fx: float, Fy: float) -> None:
        """Compute loss vector (Fx,Fy) at step t and log it."""
        self.loss = np.array([self.desired_Fx_in_t[t] - Fx,
                              self.desired_Fy_in_t[t] - Fy], dtype=np.float32)
        self.loss = self.loss / Variabs.norm_force
        self.loss_in_t[t, : self.loss.shape[0]] = self.loss
        self.loss_MSE = np.sqrt(np.sum(self.loss**2))
        self.loss_MSE_in_t[t] = self.loss_MSE

    def calc_update_tip(self, t: int, Strctr: "StructureClass", Variabs: "VariablesClass", State: "StateClass",
                        current_tip_pos: Optional[np.ndarray] = None,
                        prev_tip_update_pos: Optional[np.ndarray] = None,
                        current_tip_angle: Optional[float] = None,
                        prev_tip_update_angle: Optional[float] = None,
                        correct_for_total_angle: Optional[bool] = False,
                        correct_for_coil: Optional[bool] = True,
                        correct_for_cut_origin: Optional[bool] = True) -> None:
        """Compute next tip position/angle commands from current loss and state (pure NumPy)."""
        # Normalised inputs/outputs (NumPy)

        # --- BEASTAL or one_to_one ---
        if self.update_scheme == 'BEASTAL':
            # inputs_normalized = np.array([0, 0, 0], dtype=np.float32)
            # outputs_normalized = np.array([current_tip_pos[0]/Variabs.norm_pos, current_tip_pos[1]/Variabs.norm_pos,
            #                               current_tip_angle/Variabs.norm_angle], dtype=np.float32)
            inputs_normalized = np.array([current_tip_pos[0]/Variabs.norm_pos, current_tip_pos[1]/Variabs.norm_pos,
                                          current_tip_angle/Variabs.norm_angle], dtype=np.float32)
            outputs_normalized = np.array([State.Fx/Variabs.norm_force, State.Fy/Variabs.norm_force], dtype=np.float32)
            grad_loss_vec = learning_funcs.grad_loss_FC(Strctr.NE, inputs_normalized, outputs_normalized,
                                                        Strctr.DM, Strctr.output_nodes_arr, self.loss)
            print('grad_loss_vec', grad_loss_vec)
            update_vec = - self.alpha * np.matmul(Strctr.DM_dagger, grad_loss_vec)
            delta_tip_x = update_vec[0] * Variabs.norm_pos
            delta_tip_y = update_vec[1] * Variabs.norm_pos
            delta_angle = -update_vec[2] * Variabs.norm_angle
        elif self.update_scheme == 'radial_BEASTAL':
            if t == 1:
                prev_total_angle = 0.0
                tip_update = current_tip_pos
            else:
                prev_total_angle = self.total_angle_update_in_t[t-1]
                tip_update = self.tip_pos_update_in_t[t-1, :]
            total_angle_meas = helpers_builders._get_total_angle(current_tip_pos, 0.0, Strctr.L)
            
            loss_total_angle = helpers_builders._get_scalar_in_orthogonal_dir(self.loss, total_angle_meas)
            F_total_angle = helpers_builders._get_scalar_in_orthogonal_dir(np.array([State.Fx, State.Fy]), total_angle_meas)
            # print(f'F_total_angle{F_total_angle}')
            loss_tip = helpers_builders._get_scalar_in_orthogonal_dir(self.loss, current_tip_angle)
            F_tip_angle = helpers_builders._get_scalar_in_orthogonal_dir(np.array([State.Fx, State.Fy]), current_tip_angle)
            # print(f'F_tip_angle{F_tip_angle}')

            inputs_normalized = np.array([total_angle_meas/Variabs.norm_angle, current_tip_angle/Variabs.norm_angle],
                                         dtype=np.float32)
            # print(f'inputs_normalized={inputs_normalized}')
            outputs_normalized = np.array([F_total_angle/Variabs.norm_force, F_tip_angle/Variabs.norm_force], dtype=np.float32)
            # print(f'outputs_normalized={outputs_normalized}')

            d_total_angle = - self.alpha * 1/8*(loss_total_angle*(3*outputs_normalized[0] - outputs_normalized[1] - 2*inputs_normalized[0]) +
                                                loss_tip*(3*outputs_normalized[0] - outputs_normalized[1] - 2*inputs_normalized[1]))
            d_tip_angle = - self.alpha * 1/8*(loss_total_angle*(3*outputs_normalized[1] - outputs_normalized[0] - 2*inputs_normalized[0]) +
                                                loss_tip*(3*outputs_normalized[1] - outputs_normalized[0] - 2*inputs_normalized[1]))
            # print(f'delta_Theta{d_total_angle}')
            # print(f'delta_theta{d_tip_angle}')

            loss_thetas = np.array([loss_total_angle, loss_tip])

            grad_loss_vec = learning_funcs.grad_loss_FC(Strctr.NE, inputs_normalized, outputs_normalized,
                                                        Strctr.DM, Strctr.output_nodes_arr, loss_thetas)
            # print(f'grad_loss_vec={grad_loss_vec}')
            predicted_grad_loss_1 = (outputs_normalized[0] - inputs_normalized[0])*loss_total_angle
            predicted_grad_loss_2 = (outputs_normalized[1] - inputs_normalized[0])*loss_tip
            # print(f'predicted grad_loss_vec{np.array([predicted_grad_loss_1, predicted_grad_loss_2])}')
            # update_vec = - self.alpha * np.matmul(Strctr.DM_dagger, grad_loss_vec)
            # update_vec = np.array([-d_total_angle, -d_tip_angle])
            update_vec = np.array([-d_total_angle, -d_tip_angle]) * np.sign(total_angle_meas)
            # print(f'update_vec={update_vec}')
            # delta_tip_x = update_vec[0] * -tip_update[1]  # move in direction orthogonal to tip update
            # delta_tip_y = update_vec[0] * tip_update[0]  # move in direction orthogonal to tip update
            delta_tip_x = update_vec[0] * tip_update[1]  # move in direction orthogonal to tip update
            delta_tip_y = update_vec[0] * -tip_update[0]  # move in direction orthogonal to tip update
            # delta_angle = update_vec[1] * Variabs.norm_angle * 2
            delta_angle = update_vec[1] * Variabs.norm_angle
        elif self.update_scheme == 'BEASTAL_no_pinv':
            # large_angle = np.arctan2(self.tip_pos_int_t[t, 1], self.tip_pos_in_t[t, 0])
            # R = np.sqrt(self.tip_pos_int_t[t, 1]**2 + self.tip_pos_int_t[t, 1]**2)

            # delta_tip = - self.alpha * self.loss[:2] / Variabs.norm_force
            delta_tip_x = + self.alpha * self.loss[0] / Variabs.norm_force * Strctr.hinges * Strctr.L * (current_tip_pos[0] - 
                                                                                                         Strctr.hinges * 
                                                                                                         Strctr.L)
            delta_tip_y = - self.alpha * self.loss[1] / Variabs.norm_force * Strctr.hinges * Strctr.L * current_tip_pos[1]
            if self.loss.size == 3:
                delta_angle = + self.alpha * self.loss[2] / Variabs.norm_torque * np.pi/64 * current_tip_angle
            else:
                delta_angle = 0.0
        elif self.update_scheme == 'one_to_one':
            # large_angle = np.arctan2(self.tip_pos_int_t[t, 1], self.tip_pos_in_t[t, 0])
            # R = np.sqrt(self.tip_pos_int_t[t, 1]**2 + self.tip_pos_int_t[t, 1]**2)
            sgnx = np.sign(self.tip_pos_update_in_t[t-1, 0])
            sgny = np.sign(self.tip_pos_update_in_t[t-1, 0])
            # sgny = np.sign(self.tip_pos_update_in_t[t-1, 1])
            if sgnx == 0.0:  # sign can't be 0
                sgnx = 1
            if sgny == 0.0:
                sgny = 1
            delta_tip_y = - self.alpha * self.loss[0] * Strctr.hinges * Variabs.norm_pos * sgnx
            delta_tip_x = self.alpha * self.loss[0] * Strctr.hinges * Variabs.norm_pos * sgny
            delta_angle = - self.alpha * self.loss[1] * Variabs.norm_angle * np.pi
            print(f'delta_tip before corr {delta_tip_x},{delta_tip_y}')
            print(f'delta_angle before corr {delta_angle}')
        elif self.update_scheme == 'radial_one_to_one':
            if t == 1:
                prev_total_angle = 0.0
                tip_update = current_tip_pos
            else:
                prev_total_angle = self.total_angle_update_in_t[t-1]
                tip_update = self.tip_pos_update_in_t[t-1, :]
            # total_angle = helpers_builders._get_total_angle(tip_update, prev_total_angle, Strctr.L)
            total_angle = helpers_builders._get_total_angle(current_tip_pos, 0.0, Strctr.L)
            # print(f'total_angle_for_loss={total_angle}')
            loss_total_angle = -self.loss[0]*np.sin(total_angle) + self.loss[1]*np.cos(total_angle)
            # print(f'loss_total_angle={loss_total_angle:.2f}')
            loss_tip = -self.loss[0]*np.sin(current_tip_angle) + self.loss[1]*np.cos(current_tip_angle)
            # print(f'tip_angle_for_loss={current_tip_angle}')
            # print(f'loss_tip={loss_tip:.2f}')
            # delta_tip_x = (- self.alpha * loss_total_angle) * current_tip_pos[1]
            # delta_tip_y = (- self.alpha * loss_total_angle) * (-current_tip_pos[0])
            delta_tip_x = (- self.alpha * loss_total_angle) * tip_update[1]
            delta_tip_y = (- self.alpha * loss_total_angle) * -tip_update[0]
            delta_angle = - self.alpha * loss_tip * Variabs.norm_angle * 2
        elif self.update_scheme == 'radial_halfway_BEASTAL':
            if t == 1:
                prev_total_angle = 0.0
                tip_update = current_tip_pos
            else:
                prev_total_angle = self.total_angle_update_in_t[t-1]
                tip_update = self.tip_pos_update_in_t[t-1, :]
            # total_angle = helpers_builders._get_total_angle(tip_update, prev_total_angle, Strctr.L)
            total_angle = helpers_builders._get_total_angle(current_tip_pos, 0.0, Strctr.L)
            print(f'total_angle_for_loss={total_angle}')
            loss_total_angle = -self.loss[0]*np.sin(total_angle) + self.loss[1]*np.cos(total_angle)
            print(f'loss_total_angle={loss_total_angle:.2f}')
            loss_tip = -self.loss[0]*np.sin(current_tip_angle) + self.loss[1]*np.cos(current_tip_angle)
            print(f'tip_angle_for_loss={current_tip_angle}')
            print(f'loss_tip={loss_tip:.2f}')
            F_total_angle = helpers_builders._get_scalar_in_orthogonal_dir(np.array([State.Fx, State.Fy]), total_angle)
            print(f'F_total_angle={F_total_angle}')
            F_tip_angle = helpers_builders._get_scalar_in_orthogonal_dir(np.array([State.Fx, State.Fy]), current_tip_angle)
            print(f'F_tip_angle={F_tip_angle}')

            # delta_tip_x = (- self.alpha * loss_total_angle) * current_tip_pos[1]
            # delta_tip_y = (- self.alpha * loss_total_angle) * (-current_tip_pos[0])
            delta_tip_x = (self.alpha * loss_total_angle) * tip_update[1] * F_total_angle/Variabs.norm_force
            delta_tip_y = (self.alpha * loss_total_angle) * -tip_update[0] * F_total_angle/Variabs.norm_force
            delta_angle = self.alpha * loss_tip * Variabs.norm_angle * 2 * F_tip_angle/Variabs.norm_force

        # elif self.update_scheme == 'one_to_one_2D':
        #     # large_angle = np.arctan2(self.tip_pos_int_t[t, 1], self.tip_pos_in_t[t, 0])
        #     # R = np.sqrt(self.tip_pos_int_t[t, 1]**2 + self.tip_pos_int_t[t, 1]**2)

        #     # delta_tip = - self.alpha * self.loss[:2] / Variabs.norm_force
        #     delta_tip_x = + self.alpha * self.loss[0] / Variabs.norm_force * Strctr.hinges * Strctr.L
        #     delta_tip_y = - self.alpha * self.loss[1] / Variabs.norm_force * Strctr.hinges * Strctr.L
        #     delta_tip = np.array([delta_tip_x, delta_tip_y])
        #     delta_angle = + self.alpha * self.loss[2] / Variabs.norm_torque * np.pi/64 if (self.loss.size == 3) else 0.0
        #     print('delta_tip=', delta_tip)
        #     print('delta_angle=', delta_angle)
        else:
            raise ValueError(f"Unknown update_scheme='{self.update_scheme}'")
        delta_tip = np.array([delta_tip_x, delta_tip_y])
        # print('delta_tip=', delta_tip)
        # print('delta_angle=', delta_angle)

        if self.normalize_step and np.linalg.norm(np.append(delta_tip, delta_angle)) > 10**(-12):  # normalize if non-zero update
            step_size = np.linalg.norm(np.append(delta_tip, delta_angle))
            # print(f'step_size={step_size}')
            tradeoff_pos_angle = 1/2
            delta_tip = copy.copy(delta_tip)/step_size*self.alpha
            delta_angle = copy.copy(delta_angle)/step_size*self.alpha * tradeoff_pos_angle
            # step_size = 1
            # self.tip_pos_update_in_t[t, :] = prev_tip_update_pos + delta_tip
            # print(f'delta_tip after normalization={delta_tip}')
            # print(f'delta_angle after normalization={delta_angle}')
            # self.tip_pos_update_in_t[t, :] = prev_tip_update_pos + self.alpha*delta_tip/step_size
            # self.tip_angle_update_in_t[t] = prev_tip_update_angle + self.alpha*(float(delta_angle) + delta_total_angle)/step_size
            # print(f'normalized position step from {delta_tip} to {self.alpha*delta_tip/step_size}')
            # print(f'normalized angle step from {float(delta_angle) + delta_total_angle}')
            # print(f'to {self.alpha*(float(delta_angle) + delta_total_angle)/step_size}')

        # insert into tip_pos_update
        if prev_tip_update_pos is None:
            prev_tip_update_pos = self.tip_pos_update_in_t[t-1, :]
        # delta_tip = self.alpha*(np.array([Fx, Fy]) - current_tip_pos)*(self.loss) * ([2, 0.5])  # not BEASTAL
        # print(f'prev_tip_update_pos{prev_tip_update_pos}')
        # print(f'delta_tip{delta_tip}')
        self.tip_pos_update_in_t[t, :] = prev_tip_update_pos + delta_tip

        # Angle update (only if enabled)
        if prev_tip_update_angle is None and t > 0:
            prev_tip_update_angle = float(self.tip_angle_update_in_t[t - 1])
        elif prev_tip_update_angle is None:
            prev_tip_update_angle = current_tip_angle
        self.tip_angle_update_in_t[t] = prev_tip_update_angle + float(delta_angle)

        # add change in tip angle to the total angle from the origin
        if correct_for_total_angle:
            if t == 1:
                prev_total_angle = 0.0
            else:
                prev_total_angle = self.total_angle_update_in_t[t-1]
            # print('prev_total_angle', prev_total_angle)
            total_angle = helpers_builders._get_total_angle(self.tip_pos_update_in_t[t, :], prev_total_angle, Strctr.L)
            self.total_angle_update_in_t[t] = total_angle
            delta_total_angle = total_angle - prev_total_angle
            # print('total_angle', total_angle)
            # print('delta_total_angle', delta_total_angle)
            self.tip_angle_update_in_t[t] += delta_total_angle

        # print('update_tip_y', self.tip_pos_update_in_t[t][1])
        # print('update angle', self.tip_angle_update_in_t[t])
        # print('update angle change', self.tip_angle_update_in_t[t]-self.tip_angle_update_in_t[t-1])

        # correct for to big a stretch
        # self.tip_pos_update_in_t[t, :] = helpers_builders._correct_big_stretch(self.tip_pos_update_in_t[t],
        #                                                                        self.tip_angle_update_in_t[t], total_angle,
        #                                                                        Strctr.L, Strctr.edges)
        self.tip_pos_update_in_t[t, :] = helpers_builders._correct_big_stretch_robot_style(self.tip_pos_update_in_t[t], 
                                                                                           self.tip_angle_update_in_t[t],
                                                                                           total_angle, self.R_free, Strctr.L,
                                                                                           margin=0.1)

        cond_coil = np.abs(self.tip_angle_update_in_t[t]) > 3.5*np.pi
        cond_cut_origin = np.linalg.norm(self.tip_pos_update_in_t[t, :]) < Strctr.L
        if (correct_for_coil and cond_coil) or (correct_for_cut_origin and cond_cut_origin):
            if cond_coil:
                print('coiled up too much')
            if cond_cut_origin:
                print('cut origin')
            self.tip_pos_update_in_t[t, :] = self.tip_pos_in_t[t, :]
            print(f'setting update tip position as{self.tip_pos_update_in_t[t, :]}')
            self.tip_angle_update_in_t[t] = self.tip_angle_in_t[t]
            print(f'setting update tip angle as{self.tip_angle_update_in_t[t]}')

        delta_tip_after_corr = self.tip_pos_update_in_t[t, :] - self.tip_pos_update_in_t[t-1, :]
        delta_angle_after_corr = self.tip_angle_update_in_t[t] - self.tip_angle_update_in_t[t-1]
        print(f'delta_tip after corr {delta_tip_after_corr}')
        print(f'delta_angle after corr {delta_angle_after_corr}')

        # if correct_for_cut_origin:
            # self.tip_pos_update_in_t[t] = helpers_builders.clamp_tip_no_cross(self.tip_pos_update_in_t[t, :],
            #                                                                   self.tip_angle_update_in_t[t], Strctr.L)

    # def clamp_to_circle_xy(self, Strctr: "StructureClass", tip_pos_update, tip_angle, margin=2.0):
    #     """
    #     If (x,y) is outside the circle of radius (R-margin), project it to the nearest point on the circle.
    #     """
    #     # account for previous total angle to calculate current total angle, in [deg]

    #     # effective radius of chain
    #     R_eff = helpers_builders.effective_radius(self.R_chain, self.L, self.total_angle, tip_angle)
    #     print(f'effective Radius inside clamp_to_circle_xy = {R_eff}')

    #     r_chain = np.hypot(tip_pos_update[0], tip_pos_update[1])

    #     x2, y2 = None, None, None, None

    #     if r_chain >= (R_eff - margin):
    #         scale = (R_eff - margin) / r_chain
    #         x2 = tip_pos_update[0] * scale
    #         y2 = tip_pos_update[1] * scale
    #         print(f'clamped from x={tip_pos_update[0]},y={tip_pos_update[1]} to x={x2},y={y2} due to chain revolusions')

    #     x_clamp = np.nanmin(np.array([tip_pos_update[0], x2], dtype=float))
    #     y_clamp = np.nanmin(np.array([tip_pos_update[1], y2], dtype=float))

    #     return float(x_clamp), float(y_clamp)
