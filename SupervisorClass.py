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


# ===================================================
# Class - Supervisor Variables - training set, losses, etc.
# ===================================================


class SupervisorClass:
    """
    Variables that are by the external supervisor in the experiment

    alpha : float
        Step size for updating the commanded tip pose.
    T : int
        Number of training steps in the dataset.
    desired_buckle_arr : ndarray[int]
        Desired buckle configuration (H,S).
    sampling : str, optional
        Method for generating the command dataset. One of:
        - 'uniform'       = random uniform vals for x, y, angle
        - 'almost flat'   = flat piece, single measurement
        - 'stress strain' = immitate stress strain where compression is in x axis, incremental
    control_tip_angle : bool, default=True
        If True, tip angle is controlled and included in losses/updates.
        If False, only tip position is controlled.
    control_first_edge : bool, default=True
        If True, nodes 0 and 1 are fixed.  If False, only node 0 is fixed.
    update_scheme : str, How tip commands are updated from the loss:
        - 'one_to_one'      = direct normalized loss, equal to num of outputs
        - 'BEASTAL'         = update using pseudoinverse of the incidence matrix.
        - 'BEASTAL_no_pinv' = update using (y_j)(Loss_j), no psuedo inv of the incidence matrix.
    loss_type : str, Selects which physical quantities appear in the loss vector.
        -'cartesian'          = uses (Fx, Fy, tip_torque (if tip angle is controlled)).
        - 'Fx_and_tip_torque' = uses (Fx, tip_torque (if tip angle is controlled))
    """  
    # --- configuration / hyperparams ---
    T: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    update_scheme: str = eqx.field(static=True)
    control_tip_angle: bool = eqx.field(static=True)
    control_first_edge: bool = eqx.field(static=True)

    # --- desired targets (fixed-size buffers; NumPy, mutable at runtime) ---
    desired_buckle_arr: NDArray[np.int32] = eqx.field(static=True)                 # (hinges,)
    desired_pos_in_t: NDArray[np.float32] = eqx.field(init=False, static=True)     # (nodes, 2, T)
    desired_Fx_in_t: NDArray[np.float32] = eqx.field(init=False, static=True)      # (T,)
    desired_Fy_in_t: NDArray[np.float32] = eqx.field(init=False, static=True)      # (T,)
    desired_tau_in_t: Optional[NDArray[np.float32]] = eqx.field(default=None, init=False, static=True)  # (T,)

    # --- dataset inputs (what tip we command at each step) ---
    tip_pos_in_t: NDArray[np.float32] = eqx.field(init=False, static=True)         # (T, 2)
    tip_angle_in_t: Optional[NDArray[np.float32]] = eqx.field(default=None, init=False, static=True)    # (T,)

    # --- running logs / losses ---
    loss_in_t: NDArray[np.float32] = eqx.field(init=False, static=True)            # (T, 2) or (T, 3)
    tip_pos_update_in_t: NDArray[np.float32] = eqx.field(init=False, static=True)  # (T, 2)
    tip_angle_update_in_t: Optional[NDArray[np.float32]] = eqx.field(default=None, init=False, 
                                                                     static=True)  # (T,)
    total_angle_update_in_t: Optional[NDArray[np.float32]] = eqx.field(default=None, init=False, 
                                                                       static=True)  # (T,)

    # ------ for equilibrium calculation, jax arrays ------    
    imposed_mask: jax.ndarray[bool] = eqx.field(static=True)                       # (2*nodes,)

    # --- scratch (most recent loss vector) ---
    loss: NDArray[np.float32] = eqx.field(init=False, static=True)                 # (1,), (2,) or (3,) 

    def __init__(self, Strctr, CFG) -> None:
        self.T = int(CFG.Train.T)  # total training-set size (& algorithm time, not to confuse with time to equilib state)
        self.alpha = float(CFG.Train.alpha)
        self.update_scheme = str(CFG.Train.update_scheme)
        self.control_tip_pos = bool(CFG.Train.control_tip_pos)
        self.control_tip_angle = bool(CFG.Train.control_tip_angle)
        self.control_first_edge = bool(CFG.Train.control_first_edge)  # if true, fix nodes (0, 1), else fix only node (0)

        # for equilibrium
        self.imposed_mask = self._build_imposed_mask(Strctr, self.control_tip_pos, self.control_tip_angle)

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
        if self.control_tip_angle:
            self.desired_tau_in_t = np.zeros((self.T), dtype=np.float32)

        # Dataset (commands)
        self.tip_pos_in_t = np.zeros((self.T, 2), dtype=np.float32)
        if self.control_tip_angle:
            self.tip_angle_in_t = np.zeros((self.T,), dtype=np.float32)

        # Logs / updates
        if self.control_tip_angle and CFG.Train.loss_type == 'cartesian':
            loss_size = 2
        elif not self.control_tip_angle and CFG.Train.loss_type == 'Fx_and_tip_torque':
            loss_size = 1
        else:
            loss_size = 2
        self.loss_in_t = np.zeros((self.T, loss_size), dtype=np.float32)

        self.loss_type = CFG.Train.loss_type
        # Last loss vector (shape matches control mode)
        self.loss = np.zeros(loss_size, dtype=np.float32)

        self.tip_pos_update_in_t = np.zeros((self.T, 2), dtype=np.float32)
        if self.control_tip_angle:
            self.tip_angle_update_in_t = np.zeros((self.T,), dtype=np.float32)
            self.total_angle_update_in_t = np.zeros((self.T,), dtype=np.float32)

    def _build_imposed_mask(self, Strctr: "StructureClass", control_tip_pos: bool = True, control_tip_angle: bool = True):
        n_coords = Strctr.n_coords  # 2 * nodes
        N = Strctr.nodes  # number of nodes
        last = N - 1

        # --- fixed and imposed DOFs initialize --- 
        imposed_mask = jnp.zeros((n_coords,), dtype=bool)

        # -------- imposed tip position ----------
        if control_tip_pos:
            # set tip indices as true 
            idxs = jnp.array([helpers_builders.dof_idx(last, 0), helpers_builders.dof_idx(last, 1)])
            if control_tip_angle:
                before_last_idxs = jnp.array([helpers_builders.dof_idx(last - 1, 0), helpers_builders.dof_idx(last - 1, 1)])
                idxs = jnp.concatenate([before_last_idxs, idxs])
            imposed_mask = imposed_mask.at[idxs].set(True)  
        else:
            if control_tip_angle:
                print("no tip angle could be imposed without tip loc, skipping tip angle")

        return imposed_mask

    def create_dataset(self, Strctr: "StructureClass", CFG, sampling: str, tip_pos: Optional[NDArray] = None, tip_angle = Optional[float],
                       dist_noise: float = 0.0, angle_noise: float = 0.0) -> None:
        # save as variable
        self.dataset_sampling = sampling

        # tip positions and angles for specified tip dataset
        if sampling == 'uniform':
            np.random.seed(CFG.Train.rand_key_dataset)
            x_pos_in_t = np.random.uniform((Strctr.edges-1)*Strctr.L, Strctr.edges*Strctr.L, size=self.T)
            y_pos_in_t = np.random.uniform(-Strctr.L/3, Strctr.L/3, size=self.T)
            self.tip_pos_in_t = np.stack(((x_pos_in_t), (y_pos_in_t.T)), axis=1)
            if self.control_tip_angle and self.tip_angle_in_t is not None:
                self.tip_angle_in_t[:] = np.random.uniform(-np.pi / 5, np.pi / 5, size=self.T).astype(np.float32)
        elif sampling == 'flat':
            end = float(Strctr.edges)
            tip_pos = np.array([end, 0], dtype=np.float32)
            self.tip_pos_in_t[:] = np.tile(tip_pos, (self.T, 1))
            if self.control_tip_angle and self.tip_angle_in_t is not None:
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
            if self.control_tip_angle and self.tip_angle_in_t is not None:
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

            if self.control_tip_angle and self.tip_angle_in_t is not None:
                self.tip_angle_in_t[:] = angle_noise
        else:
            raise ValueError(f"Incompatible sampling='{sampling}'")

    def set_desired(self, pos_arr: jax.Array, Fx: float, Fy: float, t: int, tau: Optional[float] = None) -> None:
        """Store ground-truth targets for step t."""
        self.desired_pos_in_t[:, :, t] = helpers_builders.jax2numpy(pos_arr)
        self.desired_Fx_in_t[t] = float(Fx)
        self.desired_Fy_in_t[t] = float(Fy)
        if self.control_tip_angle and self.desired_tau_in_t is not None and tau is not None:
            self.desired_tau_in_t[t] = float(tau)

    def calc_loss(self, Variabs: "VariablesClass", t: int, Fx: float, Fy: float, tau: Optional[float] = None) -> None:
        """Compute loss vector (Fx,Fy[,tau]) at step t and log it."""
        if self.loss_type == 'cartesian':
            # if self.control_tip_angle and tau is not None and self.desired_tau_in_t is not None:
            #     self.loss = np.array([self.desired_Fx_in_t[t] - Fx, self.desired_Fy_in_t[t] - Fy,
            #                           self.desired_tau_in_t[t] - tau], dtype=np.float32)
            #     self.loss = self.loss / np.array([Variabs.norm_force, Variabs.norm_force, 
            #                                       Variabs.norm_torque], dtype=np.float32)
            # else:
            self.loss = np.array([self.desired_Fx_in_t[t] - Fx,
                                  self.desired_Fy_in_t[t] - Fy], dtype=np.float32)
            self.loss = self.loss / Variabs.norm_force
        elif self.loss_type == 'Fx_and_tip_torque':
            if self.control_tip_angle and tau is not None and self.desired_tau_in_t is not None:
                self.loss = np.array([self.desired_Fx_in_t[t] - Fx, self.desired_tau_in_t[t] - tau], dtype=np.float32)
                # normalize, dimless
                self.loss = self.loss / np.array([Variabs.norm_force, Variabs.norm_torque], dtype=np.float32)
            else:
                self.loss = np.array([self.desired_Fx_in_t[t] - Fx], dtype=np.float32)
                # normalize, dimless
                self.loss = self.loss / np.array([Variabs.norm_force], dtype=np.float32)
        self.loss_in_t[t, : self.loss.shape[0]] = self.loss

    def calc_update_tip(self, t: int, Strctr: "StructureClass", Variabs: "VariablesClass", State: "StateClass",
                        current_tip_pos: Optional[np.ndarray] = None,
                        prev_tip_update_pos: Optional[np.ndarray] = None,
                        current_tip_angle: Optional[float] = None,
                        prev_tip_update_angle: Optional[float] = None,
                        correct_for_total_angle: Optional[bool] = False) -> None:
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
            delta_angle = -update_vec[2] * Variabs.norm_angle if self.control_tip_angle else 0.0
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
            if self.control_tip_angle and self.loss.size == 3:
                delta_angle = + self.alpha * self.loss[2] / Variabs.norm_torque * np.pi/64 * current_tip_angle
            else:
                delta_angle = 0.0
        elif self.update_scheme == 'one_to_one':
            # large_angle = np.arctan2(self.tip_pos_int_t[t, 1], self.tip_pos_in_t[t, 0])
            # R = np.sqrt(self.tip_pos_int_t[t, 1]**2 + self.tip_pos_int_t[t, 1]**2)

            # delta_tip = - self.alpha * self.loss[:2] / Variabs.norm_force
            # delta_tip_x = + self.alpha * self.loss[0] / Variabs.norm_force * Strctr.hinges * Strctr.L
            delta_tip_x = 0.0
            if self.loss_type == 'cartesian':
                # delta_tip_y = - self.alpha * self.loss[1] / Variabs.norm_force * Strctr.hinges * Strctr.L
                # delta_angle = + self.alpha * self.loss[2] / Variabs.norm_torque * np.pi/64 if (self.control_tip_angle and 
                #                                                                                self.loss.size == 3) else 0.0
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
            elif self.loss_type == 'Fx_and_tip_torque':
                # norm_y = Variabs.norm_force * Strctr.hinges * Strctr.L
                # norm_angle = Variabs.norm_torque * np.pi if (self.control_tip_angle and self.loss.size == 2) else 0.0
                # delta_tip_y = - self.alpha * current_tip_pos[1] / Strctr.L * self.loss[0] / norm_y
                # delta_angle = - self.alpha * current_tip_angle / (2*np.pi) * self.loss[1] / norm_angle
                # delta_angle = - self.alpha * current_tip_angle * self.loss[1] * Variabs.norm_angle
                # delta_tip_y = - self.alpha * np.sign(current_tip_pos[1]) * self.loss[0] * norm_y
                # delta_angle = - self.alpha * np.sign(current_tip_angle) * self.loss[1] * norm_angle
                delta_tip_y = - self.alpha * self.loss[0] * Strctr.hinges * Variabs.norm_pos
                delta_tip_x = copy.copy(delta_tip_y)
                delta_angle = - self.alpha * self.loss[1] * Variabs.norm_angle * np.pi
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
        #     delta_angle = + self.alpha * self.loss[2] / Variabs.norm_torque * np.pi/64 if (self.control_tip_angle and 
        #                                                                                    self.loss.size == 3) else 0.0
        #     print('delta_tip=', delta_tip)
        #     print('delta_angle=', delta_angle)
        else:
            raise ValueError(f"Unknown update_scheme='{self.update_scheme}'")
        delta_tip = np.array([delta_tip_x, delta_tip_y])
        # print('delta_tip=', delta_tip)
        # print('delta_angle=', delta_angle)

        # insert into tip_pos_update
        if prev_tip_update_pos is None:
            prev_tip_update_pos = self.tip_pos_update_in_t[t-1, :]
        # delta_tip = self.alpha*(np.array([Fx, Fy]) - current_tip_pos)*(self.loss) * ([2, 0.5])  # not BEASTAL
        self.tip_pos_update_in_t[t, :] = prev_tip_update_pos + delta_tip

        # Angle update (only if enabled)
        if self.control_tip_angle and self.tip_angle_update_in_t is not None:
            if prev_tip_update_angle is None and t > 0:
                prev_tip_update_angle = float(self.tip_angle_update_in_t[t - 1])
            elif prev_tip_update_angle is None:
                prev_tip_update_angle = 0.0
            self.tip_angle_update_in_t[t] = prev_tip_update_angle + float(delta_angle)

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
        self.tip_pos_update_in_t[t, :] = helpers_builders._correct_big_stretch(self.tip_pos_update_in_t[t],
                                                                               self.tip_angle_update_in_t[t], total_angle,
                                                                               Strctr.L, Strctr.edges)

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
