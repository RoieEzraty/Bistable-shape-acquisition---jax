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

    # ------ for equilibrium calculation, jax arrays ------    
    imposed_mask: jax.ndarray[bool] = eqx.field(static=True)                       # (2*nodes,)

    # --- scratch (most recent loss vector) ---
    loss: NDArray[np.float32] = eqx.field(init=False, static=True)                 # (1,), (2,) or (3,) 

    def __init__(self, Strctr, alpha: float, T: int, desired_buckle_arr: np.ndarray, sampling='Uniform',
                 control_tip_pos: bool = True, control_tip_angle: bool = True, control_first_edge: bool = True,
                 update_scheme: str = 'one_to_one', loss_type: str = 'cartesian') -> None:
        
        self.T = int(T)  # total training-set size (& algorithm time, not to confuse with time to equilib state)
        self.alpha = float(alpha)
        self.update_scheme = str(update_scheme)
        self.control_tip_angle = bool(control_tip_angle)
        self.control_first_edge = bool(control_first_edge)  # if true, fix nodes (0, 1), else fix only node (0)

        # for equilibrium
        self.imposed_mask = self._build_imposed_mask(Strctr, control_tip_pos, control_tip_angle)

        # Desired/targets
        self.desired_buckle_arr = np.asarray(desired_buckle_arr, dtype=np.int32)
        self.desired_pos_in_t = np.zeros((Strctr.nodes, 2, T), dtype=np.float32)
        self.desired_Fx_in_t = np.zeros((T), dtype=np.float32)
        self.desired_Fy_in_t = np.zeros((T), dtype=np.float32)
        if control_tip_angle:
            self.desired_tau_in_t = np.zeros((T), dtype=np.float32)

        # Dataset (commands)
        self.tip_pos_in_t = np.zeros((self.T, 2), dtype=np.float32)
        if self.control_tip_angle:
            self.tip_angle_in_t = np.zeros((self.T,), dtype=np.float32)

        # Logs / updates
        if self.control_tip_angle and loss_type == 'cartesian':
            loss_size = 3
        elif not self.control_tip_angle and loss_type == 'Fx_and_tip_torque':
            loss_size = 1
        else:
            loss_size = 2
        self.loss_in_t = np.zeros((self.T, loss_size), dtype=np.float32)

        self.loss_type = loss_type
        # Last loss vector (shape matches control mode)
        self.loss = np.zeros(loss_size, dtype=np.float32)

        self.tip_pos_update_in_t = np.zeros((self.T, 2), dtype=np.float32)
        if self.control_tip_angle:
            self.tip_angle_update_in_t = np.zeros((self.T,), dtype=np.float32)

    def _build_imposed_mask(self, Strctr: "StructureClass", control_tip_pos: bool = True, control_tip_angle: bool = True):
        n_coords = Strctr.n_coords
        N = Strctr.hinges + 2                          # number of nodes
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

    def create_dataset(self, Strctr: "StructureClass", sampling: str, exp_start: float = None,
                       distance: float = None, dist_noise: float = 0.0, angle_noise: float = 0.0) -> None:
        # save as variable
        self.dataset_sampling = sampling

        # tip positions and angles for specified tip dataset
        if sampling == 'uniform':
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
        elif sampling == 'stress strain':
            start = 2*Strctr.L + exp_start
            end = start - distance
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
        self.desired_pos_in_t[:, :, t] = helpers_builders.numpify(pos_arr)
        self.desired_Fx_in_t[t] = float(Fx)
        self.desired_Fy_in_t[t] = float(Fy)
        if self.control_tip_angle and self.desired_tau_in_t is not None and tau is not None:
            self.desired_tau_in_t[t] = float(tau)

    def calc_loss(self, t: int, Fx: float, Fy: float, tau: Optional[float] = None) -> None:
        """Compute loss vector (Fx,Fy[,tau]) at step t and log it."""
        if self.loss_type == 'cartesian':
            if self.control_tip_angle and tau is not None and self.desired_tau_in_t is not None:
                self.loss = np.array(
                    [self.desired_Fx_in_t[t] - Fx, self.desired_Fy_in_t[t] - Fy, self.desired_tau_in_t[t] - tau],
                    dtype=np.float32,
                )
            else:
                self.loss = np.array(
                    [self.desired_Fx_in_t[t] - Fx, self.desired_Fy_in_t[t] - Fy],
                    dtype=np.float32,
                )
        elif self.loss_type == 'Fx_and_tip_torque':
            if self.control_tip_angle and tau is not None and self.desired_tau_in_t is not None:
                self.loss = np.array([self.desired_Fx_in_t[t] - Fx, self.desired_tau_in_t[t] - tau], dtype=np.float32)
            else:
                self.loss = np.array([self.desired_Fx_in_t[t] - Fx], dtype=np.float32)
        self.loss_in_t[t, : self.loss.shape[0]] = self.loss

    def calc_update_tip(self, t: int, Strctr: "StructureClass", Variabs: "VariablesClass", State: "StateClass",
                        current_tip_pos: Optional[np.ndarray] = None,
                        prev_tip_update_pos: Optional[np.ndarray] = None,
                        current_tip_angle: Optional[float] = None,
                        prev_tip_update_angle: Optional[float] = None,) -> None:
        """Compute next tip position/angle commands from current loss and state (pure NumPy)."""
        # Normalised inputs/outputs (NumPy)
        inputs_normalized = np.array([self.tip_pos_in_t[t][0]/Variabs.norm_pos, self.tip_pos_in_t[t][1]/Variabs.norm_pos,
                                      self.tip_angle_in_t[t]/Variabs.norm_angle], dtype=np.float32)
        outputs_normalized = np.array([State.Fx/Variabs.norm_force, State.Fy/Variabs.norm_force,
                                       State.tip_torque/Variabs.norm_torque], dtype=np.float32)

        # --- BEASTAL or one_to_one ---
        if self.update_scheme == 'BEASTAL':
            grad_loss_vec = learning_funcs.grad_loss_FC(Strctr.NE, inputs_normalized, outputs_normalized,
                                                        Strctr.DM, Strctr.output_nodes_arr, self.loss)

            update_vec = - self.alpha * np.matmul(Strctr.DM_dagger, grad_loss_vec)
            delta_tip = update_vec[0:2]
            delta_angle = update_vec[2] if self.control_tip_angle else 0.0

        elif self.update_scheme == 'BEASTAL_no_pinv':
            # large_angle = np.arctan2(self.tip_pos_int_t[t, 1], self.tip_pos_in_t[t, 0])
            # R = np.sqrt(self.tip_pos_int_t[t, 1]**2 + self.tip_pos_int_t[t, 1]**2)

            # delta_tip = - self.alpha * self.loss[:2] / Variabs.norm_force
            delta_tip_x = + self.alpha * self.loss[0] / Variabs.norm_force * Strctr.hinges * Strctr.L * (current_tip_pos[0] - 
                                                                                                         Strctr.hinges * 
                                                                                                         Strctr.L)
            delta_tip_y = - self.alpha * self.loss[1] / Variabs.norm_force * Strctr.hinges * Strctr.L * current_tip_pos[1]
            delta_tip = np.array([delta_tip_x, delta_tip_y])
            if self.control_tip_angle and self.loss.size == 3:
                delta_angle = + self.alpha * self.loss[2] / Variabs.norm_torque * np.pi/64 * current_tip_angle
            else:
                delta_angle = 0.0
            print('delta_tip=', delta_tip)
            print('delta_angle=', delta_angle)
        elif self.update_scheme == 'one_to_one':
            # large_angle = np.arctan2(self.tip_pos_int_t[t, 1], self.tip_pos_in_t[t, 0])
            # R = np.sqrt(self.tip_pos_int_t[t, 1]**2 + self.tip_pos_int_t[t, 1]**2)

            # delta_tip = - self.alpha * self.loss[:2] / Variabs.norm_force
            delta_tip_x = + self.alpha * self.loss[0] / Variabs.norm_force * Strctr.hinges * Strctr.L
            if self.loss_type == 'cartesian':
                delta_tip_y = - self.alpha * self.loss[1] / Variabs.norm_force * Strctr.hinges * Strctr.L
                delta_angle = + self.alpha * self.loss[2] / Variabs.norm_torque * np.pi/64 if (self.control_tip_angle and 
                                                                                               self.loss.size == 3) else 0.0
            elif self.loss_type == 'Fx_and_tip_torque':
                delta_tip_y = + self.alpha * self.loss[0] / Variabs.norm_force * Strctr.hinges * Strctr.L
                delta_angle = - self.alpha * self.loss[1] / Variabs.norm_torque * np.pi/360 if (self.control_tip_angle and 
                                                                                                self.loss.size == 2) else 0.0
            delta_tip = np.array([delta_tip_x, delta_tip_y])
            
            print('delta_tip=', delta_tip)
            print('delta_angle=', delta_angle)
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

        # insert into tip_pos_update
        if prev_tip_update_pos is None:
            prev_tip_update_pos = self.tip_pos_update_in_t[t-1, :]
        # delta_tip = self.alpha*(np.array([Fx, Fy]) - current_tip_pos)*(self.loss) * ([2, 0.5])  # not BEASTAL
        self.tip_pos_update_in_t[t, :] = prev_tip_update_pos + delta_tip
        print('prev_tip_update_pos=', prev_tip_update_pos)
        print('tip_pos_update_in_t[t, :]=', self.tip_pos_update_in_t[t, :])

        # Angle update (only if enabled)
        if self.control_tip_angle and self.tip_angle_update_in_t is not None:
            if prev_tip_update_angle is None and t > 0:
                prev_tip_update_angle = float(self.tip_angle_update_in_t[t - 1])
            elif prev_tip_update_angle is None:
                prev_tip_update_angle = 0.0
            self.tip_angle_update_in_t[t] = float(prev_tip_update_angle + float(delta_angle))

        # correct for to big a stretch
        self.tip_pos_update_in_t[t, :] = helpers_builders._correct_big_stretch(self.tip_pos_update_in_t[t],
                                                                               self.tip_angle_update_in_t[t], Strctr.L,
                                                                               Strctr.edges)
