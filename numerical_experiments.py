import numpy as np
import matplotlib.pyplot as plt
import time

from numpy.typing import NDArray
from typing import TYPE_CHECKING, Tuple, Optional

import plot_funcs, file_funcs

from StructureClass import StructureClass
from VariablesClass import VariablesClass
from StateClass import StateClass
from SupervisorClass import SupervisorClass
from EquilibriumClass import EquilibriumClass
from config import ExperimentConfig


# ---------------------------------------------------------------
# Full training
# ---------------------------------------------------------------
def train(Strctr: StructureClass, Variabs: VariablesClass, Sprvsr: SupervisorClass, State_meas: StateClass,
          State_update: StateClass, State_des: StateClass, CFG: ExperimentConfig):
    """
    Run the closed-loop training loop: measure → compute loss → update tip command → relax → buckle.

    At each training step `t`, this routine performs two equilibrium solves under the *same* commanded
    tip pose (position + angle):
      1) **Measurement**: equilibrate using the *current* buckle state (`State_meas.buckle_arr`).
      2) **Desired**: equilibrate using the *target* buckle configuration (`Sprvsr.desired_buckle_arr`).
      3) **Loss and Update**: compute loos and equilibrate again with updated tip command (`Sprvsr.tip_pos_update_in_t[t]`,
         `Sprvsr.tip_angle_update_in_t[t]`) and the current update buckle state (`State_update.buckle_arr`).
      4) Apply the **buckling rule** (`State_update.buckle(...)`) to flip shims whose hinge angles
         cross thresholds, and mirror those flips into `State_meas`.

    Notes
    -----
    - Loop starts at `t=1` (step 0 assumed to be initialization, already stored in the states).

    Parameters
    ----------
    State_meas   - StateClass. "Measured" modality state (current buckle), updated each step from equilibrium.
    State_update - StateClass. "Update" modality state (after applying the learned command update), buckles may flip.
    State_des    - StateClass. "Desired" modality state (target buckle config), used to generate force targets.

    Returns
    -------
    pos_in_t_meas   : ndarray, shape (T, nodes, 2) Measurement modality equilibrium node positions over training steps.
    pos_in_t_update : ndarray, shape (T, nodes, 2) Update modality equilibrium node positions over training steps.
    """
    for t in range(1, Sprvsr.T):    
        print('t=', t)   
        
        ## MEASUREMENT
        print('===MEASUREMENT===')
        
        tip_pos = Sprvsr.tip_pos_in_t[t]
        tip_angle = Sprvsr.tip_angle_in_t[t] 
        # tip_pos = np.array([0.24, 0.033769])
        # tip_angle = 0.54186946
        # print('tip_pos=', tip_pos)
        # print('tip_angle=', tip_angle)
        
        init_tip_update_pos = np.array([Strctr.L*Strctr.edges, 0.0])
        init_tip_update_angle = 0.0
        
        # --- equilibrium - measured & desired---
        Eq_meas = EquilibriumClass(Strctr, CFG, buckle_arr=State_meas.buckle_arr, pos_arr=State_meas.pos_arr)  # meausrement
        Eq_des = EquilibriumClass(Strctr, CFG, buckle_arr=Sprvsr.desired_buckle_arr, pos_arr=State_des.pos_arr)  # desired
        final_pos, pos_in_t, _, F_theta = Eq_meas.calculate_state(Variabs, Strctr, Sprvsr, init_pos=None,
                                                                  tip_pos=tip_pos, tip_angle=tip_angle)
        final_pos_des, pos_in_t_des, _, F_theta_des = Eq_des.calculate_state(Variabs, Strctr, Sprvsr, init_pos=None, 
                                                                             tip_pos=tip_pos, tip_angle=tip_angle)
    #     edge_lengths = vmap(lambda e: Strctr._get_edge_length(final_pos, e))(jnp.arange(Strctr.edges))
    #     print('edge lengths', helpers_builders.numpify(edge_lengths))

        # --- save sizes and plot - measured & desired ---
        State_meas._save_data(t, Strctr, final_pos, State_meas.buckle_arr, F_theta)
        State_des._save_data(t, Strctr, final_pos_des, State_des.buckle_arr, F_theta_des)
        
        Sprvsr.set_desired(final_pos_des, State_des.Fx, State_des.Fy, t)
        plot_funcs.plot_arm(final_pos, State_meas.buckle_arr, Strctr.L, modality="measurement")
    #     print('potential F sum', F_theta)
        plot_funcs.plot_arm(final_pos_des, State_des.buckle_arr, Strctr.L, modality="measurement")
    #     print('potential F summed desired', F_theta_des)
    #     # print('Forces', potential_force_in_t[-1])
    #     print('Fx on tip, measurement', State_meas.Fx)
    #     print('Fx on tip, desired', State_des.Fx)
    #     print('Fy on tip, measurement', State_meas.Fy)
    #     print('Fy on tip, desired', State_des.Fy)
    # #     plt.plot(potential_force_in_t[-1,:], '.')
    # #     plt.show()
        
        # ------- loss ------- 
        Sprvsr.calc_loss(Variabs, t, State_meas.Fx, State_meas.Fy)
        print('desired Fx=', Sprvsr.desired_Fx_in_t[t])
        print('measured Fx=', State_meas.Fx)
        print('loss', Sprvsr.loss)
        
        # ------- UPDATE ------- 
        print('===UPDATE===')
        
        print('current_tip_pos =', tip_pos)
        print('current_tip_angle =', tip_angle)
        if t == 1:
            Sprvsr.calc_update_tip(t, Strctr, Variabs, State_meas, current_tip_pos=tip_pos, current_tip_angle=tip_angle,
                                   prev_tip_update_pos=init_tip_update_pos, prev_tip_update_angle=init_tip_update_angle,
                                   correct_for_total_angle=True)
        else:
            Sprvsr.calc_update_tip(t, Strctr, Variabs, State_meas, current_tip_pos=tip_pos, current_tip_angle=tip_angle,
                                   correct_for_total_angle=True)

        # --- equilibrium ---
        # print('init_pos_update', State_update.pos_arr_in_t[:, :, t-1])
        final_pos, pos_in_t_update, _, F_theta = Eq_meas.calculate_state(Variabs, Strctr, Sprvsr,
                                                                         State_update.pos_arr_in_t[:, :, t-1],
                                                                         tip_pos=Sprvsr.tip_pos_update_in_t[t],
                                                                         tip_angle=Sprvsr.tip_angle_update_in_t[t])

        # --- save sizes and plot ---
        State_update._save_data(t, Strctr, final_pos, State_update.buckle_arr, F_theta)
        plot_funcs.plot_arm(final_pos, State_update.buckle_arr, Strctr.L, modality="update")
    #     print('pre buckle', State_update.buckle_arr.T)
        # print('energy', Eq.energy(Variabs, Strctr, final_pos)[-1])
        
        # --- shims buckle ---
        State_update.buckle(Variabs, Strctr, t, State_measured=State_meas)      
    #     Eq = EquilibriumClass(Strctr, T_eq, damping, mass, buckle_arr=helpers_builders.jaxify(State_update.buckle_arr),
    #                           pos_arr=helpers_builders.jaxify(State_update.pos_arr))
    #     print('post buckle', State_update.buckle_arr.T)
    #     # print('post buckle update', State_update.buckle_arr)
    #     # print('energy', Eq.energy(Variabs, Strctr, final_pos)[-1])
        plot_funcs.plot_arm(final_pos, State_update.buckle_arr, Strctr.L, modality="update")

    pos_in_t_meas = np.moveaxis(State_meas.pos_arr_in_t, 2, 0)
    pos_in_t_update = np.moveaxis(State_update.pos_arr_in_t, 2, 0)

    return pos_in_t_meas, pos_in_t_update


# ---------------------------------------------------------------
# Single tip movement, incremental equilibration
# ---------------------------------------------------------------
def compress_to_tip_pos(Strctr: "StructureClass", Variabs: "VariablesClass", Sprvsr: "SupervisorClass", CFG: ExperimentConfig,
                        buckle: NDArray, tip_pos_i: NDArray, tip_angle_i: float, tip_pos_f: NDArray, tip_angle_f: float,
                        Eq_iterations: int) -> Tuple["StateClass", list[NDArray], list[NDArray]]:
    """
    Incrementally compress origami to final tip position and angle, ensuring stable convergence, in Eq_iterations steps.

    This function performs a sequence of equilibrium simulations, gradually moving the tip 
    from an initial position and angle `(tip_pos_i, tip_angle_i)` to a final configuration 
    `(tip_pos_f, tip_angle_f)`. After each intermediate step, the new equilibrium state 
    becomes the starting point for the next iteration. This progressive approach stabilizes 
    the numerical integration for stiff systems.

    Parameters
    ----------
    buckle        - NDArray, Initial buckle state of all hinges (typically ±1).
    tip_pos_i     - array-like of shape (2,), Initial position of the structure’s tip node.
    tip_angle_i   - float, Initial angular orientation of the tip node (radians).
    tip_pos_f     - array-like of shape (2,), Target final position of the structure’s tip node.
    tip_angle_f   - float, Target final tip orientation (radians).    
    Eq_iterations - int, how many equilibration steps along tip movement

    Returns
    -------
    State      - StateClass, Final equilibrium state of the structure after all compression steps.
    pos_in_t   - list[NDArray], List of position histories over all equilibrium phases.
                                Each entry corresponds to the node positions over time during one equilibrium phase.
    force_in_t - list[NDArray], List of potential force histories corresponding to each equilibrium phase.

    Notes
    -----
    - The function linearly interpolates both the tip position and angle across 
      `Eq_iterations` steps for smooth deformation.
    - The last step is performed with increased `T_eq` (×2) and damping (×3) 
      to ensure convergence to a steady-state equilibrium.
    - Each equilibrium step is executed via `one_shot()`, which performs a 
      single equilibrium calculation and visualization.
    """
    # initialize positions and forces
    pos_in_t = []
    force_in_t = []
    State = StateClass(Strctr, Sprvsr, buckle_arr=buckle)

    # interpolation fractions for the transition steps: (1/Eq_iterations, ..., 1)
    alphas = (np.arange(1, Eq_iterations + 1, dtype=float) / float(Eq_iterations))
    tip_pos_i_arr = np.asarray(tip_pos_i, dtype=float)
    tip_pos_f_arr = np.asarray(tip_pos_f, dtype=float)

    tip_pos_seq = [tip_pos_i_arr * (1.0 - a) + tip_pos_f_arr * a for a in alphas]
    tip_ang_seq = [float(tip_angle_i * (1.0 - a) + tip_angle_f * a) for a in alphas]

    # append one extra "hold" step at the *final* command
    tip_pos_seq.append(tip_pos_f_arr)
    tip_ang_seq.append(float(tip_angle_f))

    pos_init = None  # None for first shot, then last equilibrium thereafter

    for t, (tip_pos, tip_angle) in enumerate(zip(tip_pos_seq, tip_ang_seq)):
        pos_in_t_i, force_in_t_i = one_shot(Strctr, Variabs, Sprvsr, State, CFG, buckle, tip_pos, tip_angle,
                                            init_pos=pos_init, t=t)
        pos_in_t.append(pos_in_t_i)
        force_in_t.append(force_in_t_i)

        # Next init is the last equilibrium position from this shot
        # (assumes pos_in_t_i is a time series with last entry being final equilibrium)
        pos_init = pos_in_t_i[-1]

    return State, pos_in_t, force_in_t


# ---------------------------------------------------------------
# Full experiment from file
# ---------------------------------------------------------------
def measure_determined_pos_from_file(Strctr: "StructureClass", Variabs: "VariablesClass", Sprvsr: "SupervisorClass",
                                     CFG: ExperimentConfig, path: str, buckle: NDArray,
                                     stretch_factor: Optional[float] = None) -> Tuple[NDArray, NDArray, NDArray]:
    """
    tip performs prescribed trajectory from a CSV file, measure simulated tip forces. Export results to csv,

    Notes
    -----
    - Also load experimental forces stored in the same file
    - At each time, populate supervisor command histories:
       - `Sprvsr.tip_pos_in_t   = P[:, :2]`
       - `Sprvsr.tip_angle_in_t = P[:, 2]`
    - Assumes `file_funcs.load_pos_force(..., mod="arrays")` returns:
      `(T, P, F)` where `P[:, :2]` are positions, `P[:, 2]` is angle, and `F[:, 0/1]` are forces.

    Parameters
    ----------
    path            - str. Path to the CSV file containing tip poses and (optionally) forces.
    buckle          - ndarray. Initial buckle configuration for the chain, forwarded to `StateClass`
                      and to `one_shot(...)`.
    stretch_factor  - Optional[float]. Optional stretch rescaling passed to `load_pos_force`.

    Returns
    -------
    State        : StateClass
                   State instance used during replay; contains the final equilibrium state and logged histories.
    P            : ndarray, shape (T, 3)
                   Tip pose history loaded from file: columns are [x, y, angle].
    F_x_vec      : ndarray, shape (T,)
                   x-force at the tip for each commanded pose.
    F_y_vec      : ndarray, shape (T,)
                   y-force at the tip for each commanded pose.
    F_x_vec_exp  : ndarray, shape (T,)
                   Experimental x-force loaded from file (for comparison).
    F_y_vec_exp  : ndarray, shape (T,)
                   Experimental y-force loaded from file (for comparison).
    """
    # ------ load positions and optionally forces ------
    T, P, F = file_funcs.load_pos_force(path, mod="arrays", stretch_factor=stretch_factor)

    # Supervisor command histories (used elsewhere, and for export)
    Sprvsr.tip_pos_in_t = P[:, :2]
    Sprvsr.tip_angle_in_t = P[:, 2]

    # Experimental forces from file
    F_x_vec_exp = F[:, 0]
    F_y_vec_exp = F[:, 1]
    print('P', P)

    # ------ Initialize simulation -------
    # simulated forces
    n = P.shape[0]
    F_x_vec = np.zeros(n, dtype=float)
    F_y_vec = np.zeros(n, dtype=float)
    State = StateClass(Strctr, Sprvsr, buckle_arr=buckle)

    # Tip calibration offset (avoid allocating each loop)
    tip_offset = np.array([-0.003, 0.0], dtype=float)

    prev_final_pos = None  # warm-start position for next step

    # ------ loop ------
    for i in range(n):
        tip_pos = P[i, :2] + tip_offset
        tip_angle = float(P[i, 2])

        pos_traj, final_F = one_shot(Strctr, Variabs, Sprvsr, State, CFG, buckle, tip_pos, tip_angle, 
                                     init_pos=prev_final_pos, t=i)

        # Record simulated forces (State updated inside one_shot)
        F_x_vec[i] = State.Fx
        F_y_vec[i] = State.Fy

        # Warm start next iteration from last equilibrium position of this trajectory
        prev_final_pos = pos_traj[-1]

    # ------ export ------
    file_funcs.export_predetermined(Sprvsr, State)
    return State, P, F_x_vec, F_y_vec, F_x_vec_exp, F_y_vec_exp


# ---------------------------------------------------------------
# Single equilibration
# ---------------------------------------------------------------
def one_shot(Strctr: "StructureClass", Variabs: "VariablesClass", Sprvsr: "SupervisorClass", State: "StateClass",
             CFG: ExperimentConfig, buckle: NDArray, tip_pos: NDArray, tip_angle: float,
             init_pos: Optional[np.ndarray] = None, t: int = 0) -> Tuple["StateClass", NDArray, NDArray]:
    """
    Perform a single equilibrium computation and state update for the system.

    Equilibrium calculations through the `EquilibriumClass`, updates the state based on the computed 
    equilibrium configuration, and visualizes the arm configuration.

    Parameters
    ----------
    buckle    - NDArray, Array indicating the initial buckle configuration of the system (typically ±1).
    tip_pos   - NDArray, Prescribed position of the tip node(s) during the equilibrium calculation.
    tip_angle - NDArray, Prescribed angular orientation of the tip node(s).
    init_pos  - NDArray, initial position from which to compress and calculate state

    Returns
    -------
    State : StateClass
        Updated state object containing node positions, forces, and angles after equilibrium.
    pos_in_t : NDArray
        Array of node positions as a function of time during the equilibrium process.
    potential_force_in_t : NDArray
        Time evolution of potential forces during the equilibrium calculation.

    Notes
    -----
    - The function performs an initial equilibrium step with no tip movement yet.
    - It prints diagnostic information such as buckle configuration, edge lengths,
      stretch energy, torque energy, and net forces.
    - A plot of the arm configuration is displayed at the end of execution.
    """
    # ------ initialize, no tip movement yet ------
    Eq = EquilibriumClass(Strctr, CFG, buckle_arr=buckle, pos_arr=State.pos_arr)

    # ------ claculate equilibrium from ode dynamics ------
    if init_pos is None:
        final_pos, pos_in_t, vel_in_t, final_F = Eq.calculate_state(Variabs, Strctr, Sprvsr, init_pos=State.pos_arr,
                                                                    tip_pos=tip_pos, tip_angle=tip_angle)
    else:
        print('init_pose=', init_pos)
        final_pos, pos_in_t, vel_in_t, final_F = Eq.calculate_state(Variabs, Strctr, Sprvsr, init_pos=init_pos,
                                                                    tip_pos=tip_pos, tip_angle=tip_angle)
    # ------ save, plot, print ------
    State._save_data(t, Strctr, final_pos, State.buckle_arr, final_F)
    State.buckle(Variabs, Strctr, t, State_measured=State)   
    print('pos_arr', final_pos)
    print('edge len', Strctr.all_edge_lengths(State.pos_arr))
    print('total edge error', np.sum((Strctr.all_edge_lengths(State.pos_arr)-Strctr.L)**2)/(Strctr.edges*Strctr.L**2))
    plot_funcs.plot_arm(State.pos_arr, State.buckle_arr, Strctr.L, modality="measurement")
    plt.show()
    return pos_in_t, final_F


# ---------------------------------------------------------------
# Simulation of Bi-ax measurement, Harvard 2025
# ---------------------------------------------------------------
def ADMET_stress_strain(Strctr: StructureClass, Variabs: VariablesClass, Sprvsr: SupervisorClass, State: StateClass, 
                        CFG: ExperimentConfig, tip_angle: float, *, plot_every: int = 1
                        ) -> Tuple[NDArray[np.float_],   # Fx_afo_pos
                                   NDArray[np.float_],   # pos_frames  (T, nodes, 2)
                                   NDArray[np.int_],     # buckle_frames (T, hinges, shims)
                                   NDArray[np.float_],   # theta_frames (T, hinges)
                                   ]:
    """
    Run a quasi-static stress–strain protocol by sweeping the tip
    position according to `Sprvsr.tip_pos_in_t` to simulate an ADMET experiment by Roie.

    For each commanded tip position (and optionally tip angle), this function:
      1. Solves for equilibrium via `EquilibriumClass.calculate_state`.
      2. Updates the buckling state using `State_des.buckle`.
      3. Saves forces, positions, buckles, and hinge angles into pre-allocated
         NumPy arrays for fast indexed access.

    Parameters
    ----------
    buckle    - NDArray, Array indicating the initial buckle configuration of the system (typically ±1).
    tip_pos   - NDArray, Prescribed position of the tip node(s) during the equilibrium calculation.
    tip_angle - NDArray, Prescribed angular orientation of the tip node(s).
    pos_noise : float, optional
        Amplitude of initial position noise per DOF (see `EquilibriumClass.calculate_state`).
    vel_noise : float, optional
        Amplitude of initial velocity noise per DOF.
    plot_every : int, optional
        If > 0, plot the arm every `plot_every` steps (including step 0).

    Returns
    -------
    Fx_afo_pos : ndarray, shape (T,)
        Reaction force in x at the tip (or wall node) as a function of step index.
    pos_frames : ndarray, shape (T, nodes, 2)
        Nodal positions at equilibrium for each tip position
    buckle_frames : ndarray, shape (T, hinges, shims)
        Buckle configuration at equilibrium for each tip position
    theta_frames : ndarray, shape (T, hinges)
        Hinge angles at equilibrium for each tip position (radians).
    """
    # --- safety check - Supervisor tip position afo t has to be 'stress strain' ---
    if getattr(Sprvsr, "dataset_sampling", None) != "stress strain":
        raise ValueError(f"run_admet_stress_strain() can only be used when Sprvsr.dataset_sampling == "
                         f"'stress strain', but got '{Sprvsr.dataset_sampling}'.")

    # Number of ADMET steps (typically Sprvsr.T)
    T_steps: int = Sprvsr.tip_pos_in_t.shape[0]

    # Pre-allocate outputs
    Fx_afo_pos: NDArray[np.float_] = np.zeros(T_steps, dtype=np.float32)
    pos_frames: NDArray[np.float_] = np.zeros((T_steps, Strctr.nodes, 2), dtype=np.float32)
    buckle_frames: NDArray[np.int_] = np.zeros((T_steps, Strctr.hinges, Strctr.shims), dtype=np.int32)
    theta_frames: NDArray[np.float_] = np.zeros((T_steps, Strctr.hinges), dtype=np.float32)

    t0 = time.time()  # start counting time of computation
    final_pos: Optional[np.ndarray] = None

    for i, tip_pos in enumerate(Sprvsr.tip_pos_in_t):
        print(f"tip_pos[{i}] = {tip_pos}")

        # Equilibrium; use previous equilibrium as init_pos
        Eq = EquilibriumClass(Strctr, CFG, buckle_arr=State.buckle_arr,
                              pos_arr=final_pos if final_pos is not None else State.pos_arr)
        final_pos, pos_in_t, vel_in_t, potential_force_in_t = Eq.calculate_state(Variabs, Strctr, Sprvsr, init_pos=Eq.init_pos,
                                                                                 tip_pos=tip_pos, tip_angle=tip_angle)

        # Buckle
        State.buckle(Variabs, Strctr, i, State)

        # Save, plot, store
        State._save_data(t=i, Strctr=Strctr, pos_arr=final_pos, buckle_arr=State.buckle_arr, Forces=potential_force_in_t)
        print("State tip forces ", [State.Fx, State.Fy])
        print("edge lengths ", State.edge_lengths)
        if plot_every > 0 and (i % plot_every == 0):
            plot_funcs.plot_arm(State.pos_arr, State.buckle_arr, Strctr.L, modality="measurement")
        Fx_afo_pos[i] = State.Fx
        pos_frames[i, :, :] = State.pos_arr
        buckle_frames[i, :, :] = State.buckle_arr
        theta_frames[i, :] = State.theta_arr

    t1 = time.time()  # stop counting time of computation
    print(f"Total runtime: {t1 - t0:.2f} seconds")  # print time of computation

    return Fx_afo_pos, pos_frames, buckle_frames, theta_frames
