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


def train(Strctr: StructureClass, Variabs: VariablesClass, Sprvsr: SupervisorClass, State_meas: StateClass,
          State_update: StateClass, State_des: StateClass, CFG: ExperimentConfig):
    """
    comment here
    """
    for t in range(1, Sprvsr.T):    
        print('t=', t)   
        
        ## MEASUREMENT
        print('===MEASUREMENT===')
        
    #     tip_pos = Sprvsr.tip_pos_in_t[t]
    #     tip_angle = Sprvsr.tip_angle_in_t[t] 
        tip_pos = np.array([0.24, 0.033769])
        tip_angle = 0.54186946
        # print('tip_pos=', tip_pos)
        # print('tip_angle=', tip_angle)
        
        init_tip_update_pos = np.array([Strctr.L*Strctr.edges, 0.0])
        init_tip_update_angle = 0.0
        
        # --- equilibrium - measured & desired---
        Eq_meas = EquilibriumClass(Strctr, CFG, buckle_arr=State_meas.buckle_arr, pos_arr=State_meas.pos_arr)  # meausrement
        Eq_des = EquilibriumClass(Strctr, CFG, buckle_arr=Sprvsr.desired_buckle_arr, pos_arr=State_des.pos_arr)  # desired
        final_pos, pos_in_t, _, F_theta = Eq_meas.calculate_state(Variabs, Strctr, Sprvsr,
                                                                  init_pos=State_meas.pos_arr_in_t[:, :, t-1],
                                                                  tip_pos=tip_pos, tip_angle=tip_angle)
        final_pos_des, pos_in_t_des, _, F_theta_des = Eq_des.calculate_state(Variabs, Strctr, Sprvsr, 
                                                                  init_pos=State_meas.pos_arr_in_t[:, :, t-1], 
                                                                  tip_pos=tip_pos, tip_angle=tip_angle)
    #     edge_lengths = vmap(lambda e: Strctr._get_edge_length(final_pos, e))(jnp.arange(Strctr.edges))
    #     print('edge lengths', helpers_builders.numpify(edge_lengths))

        # --- save sizes and plot - measured & desired ---
        State_meas._save_data(t, Strctr, final_pos, State_meas.buckle_arr, F_theta, control_tip_angle=Sprvsr.control_tip_angle)
        State_des._save_data(t, Strctr, final_pos_des, State_des.buckle_arr, F_theta_des,
                             control_tip_angle=Sprvsr.control_tip_angle)
        
        Sprvsr.set_desired(final_pos_des, State_des.Fx, State_des.Fy, t, tau=State_des.tip_torque)
        plot_funcs.plot_arm(final_pos, State_meas.buckle_arr, State_meas.theta_arr, Strctr.L, modality="measurement")
    #     print('potential F sum', F_theta)
        plot_funcs.plot_arm(final_pos_des, State_des.buckle_arr, State_des.theta_arr, Strctr.L, modality="measurement")
    #     print('potential F summed desired', F_theta_des)
    #     # print('Forces', potential_force_in_t[-1])
    #     print('Fx on tip, measurement', State_meas.Fx)
    #     print('Fx on tip, desired', State_des.Fx)
    #     print('Fy on tip, measurement', State_meas.Fy)
    #     print('Fy on tip, desired', State_des.Fy)
    # #     plt.plot(potential_force_in_t[-1,:], '.')
    # #     plt.show()
    #     print('torque on tip, measurement', State_meas.tip_torque)
    #     print('torque on tip, desired', State_des.tip_torque)
        
        # ------- loss ------- 
        Sprvsr.calc_loss(Variabs, t, State_meas.Fx, State_meas.Fy, tau=State_meas.tip_torque)
        print('desired Fx=', Sprvsr.desired_Fx_in_t[t])
        print('measured Fx=', State_meas.Fx)
        print('desired torque=', Sprvsr.desired_tau_in_t[t])
        print('measured torque=', State_meas.tip_torque)
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
        State_update._save_data(t, Strctr, final_pos, State_update.buckle_arr, F_theta,
                                control_tip_angle=Sprvsr.control_tip_angle)
        plot_funcs.plot_arm(final_pos, State_update.buckle_arr, State_update.theta_arr, Strctr.L, modality="update")
    #     print('pre buckle', State_update.buckle_arr.T)
        # print('energy', Eq.energy(Variabs, Strctr, final_pos)[-1])
        
        # --- shims buckle ---
        State_update.buckle(Variabs, Strctr, t, State_measured=State_meas)      
    #     Eq = EquilibriumClass(Strctr, T_eq, damping, mass, buckle_arr=helpers_builders.jaxify(State_update.buckle_arr),
    #                           pos_arr=helpers_builders.jaxify(State_update.pos_arr))
    #     print('post buckle', State_update.buckle_arr.T)
    #     # print('post buckle update', State_update.buckle_arr)
    #     # print('energy', Eq.energy(Variabs, Strctr, final_pos)[-1])
        plot_funcs.plot_arm(final_pos, State_update.buckle_arr, State_update.theta_arr, Strctr.L, modality="update")

        pos_in_t_meas = np.moveaxis(State_meas.pos_arr_in_t, 2, 0)
        pos_in_t_udpate = np.moveaxis(State_update.pos_arr_in_t, 2, 0)

    return pos_in_t_meas, pos_in_t_udpate


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
        Strctr      - StructureClass, Structural definition containing geometry (hinges, edges, node layout, etc.).
        Variabs     - VariablesClass, Material and stiffness parameters for hinges, edges, and stretch elements.
        Sprvsr      - SupervisorClass, Supervisory object controlling simulation and visualization parameters.
        CFG         - user variables from config file
        buckle      - NDArray, Initial buckle state of all hinges (typically ±1).
        tip_pos_i   - array-like of shape (2,), Initial position of the structure’s tip node.
        tip_angle_i - float, Initial angular orientation of the tip node (radians).
        tip_pos_f   - array-like of shape (2,), Target final position of the structure’s tip node.
        tip_angle_f - float, Target final tip orientation (radians).        

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
    for i in range(Eq_iterations):
        
        # incrementally move positiong and angle
        tip_pos = tip_pos_i*(Eq_iterations-(i+1))/(Eq_iterations) + tip_pos_f*(i+1)/Eq_iterations
        tip_angle = tip_angle_i*(Eq_iterations-(i+1))/(Eq_iterations) + tip_angle_f*(i+1)/Eq_iterations
        
        # initial position of current step is the Equilibrium of the previous
        if i == 0:
            pos_init = None
        else:
            pos_init = pos_in_t[-1][-1]

        # calculate equilibrium of new tip pos and angle
        pos_in_t_i, force_in_t_0 = one_shot(Strctr, Variabs, Sprvsr, State, CFG, buckle, tip_pos, tip_angle,
                                            init_pos=pos_init, t=i)
        pos_in_t.append(pos_in_t_i)
        force_in_t.append(force_in_t_0)

    # final one is just some more time at same tip pos
    pos_init = pos_in_t[-1][-1]
    pos_in_t_i, force_in_t_0 = one_shot(Strctr, Variabs, Sprvsr, State, CFG, buckle, tip_pos, tip_angle, init_pos=pos_init, t=i)
    pos_in_t.append(pos_in_t_i)
    force_in_t.append(force_in_t_0)
    return State, pos_in_t, force_in_t


def measure_determined_pos_from_file(Strctr: "StructureClass", Variabs: "VariablesClass", Sprvsr: "SupervisorClass",
                                     CFG: ExperimentConfig, path: str, buckle: NDArray,
                                     stretch_factor: Optional[float] = None) -> Tuple[NDArray, NDArray, NDArray]:
    T, P, F = file_funcs.load_pos_force(path, mod="arrays", stretch_factor = stretch_factor)
    Sprvsr.tip_pos_in_t = P[:, :2]
    Sprvsr.tip_angle_in_t = P[:, 2]
    F_x_vec_exp = F[:, 0]
    F_y_vec_exp = F[:, 1]
    print('P', P)
    F_x_vec = np.zeros(np.shape(P)[0])
    F_y_vec = np.zeros(np.shape(P)[0])
    State = StateClass(Strctr, Sprvsr, buckle_arr=buckle)
    for i, pos in enumerate(P):
        if i == 0:
            # init_pos = np.array([[0., 0.],
            #                      [0.045, 0.],
            #                      [0.08891349, -0.11953123],
            #                      [0.12679164, -0.11454715],
            #                      [0.11983377, -0.15893409],
            #                      [0.1101802, -0.10281981],
            #                      [0.142, -0.071]])
            init_pos = None
        else:
            init_pos = pos_in_t[-1]
        tip_pos = pos[:2]
        tip_angle = pos[2]
        pos_in_t, final_F = one_shot(Strctr, Variabs, Sprvsr, State, CFG, buckle, tip_pos, tip_angle, init_pos=init_pos, t=i)
        F_x_vec[i], F_y_vec[i] = State.Fx, State.Fy
    file_funcs.export_stress_strain_sim(Sprvsr, F_x_vec, F_y_vec,  Strctr.L, buckle)
    return State, P, F_x_vec, F_y_vec, F_x_vec_exp, F_y_vec_exp


def one_shot(Strctr: "StructureClass", Variabs: "VariablesClass", Sprvsr: "SupervisorClass", State: "StateClass",
             CFG: ExperimentConfig, buckle: NDArray, tip_pos: NDArray, tip_angle: float,
             init_pos: Optional[np.ndarray] = None, t: int = 0) -> Tuple["StateClass", NDArray, NDArray]:
    """
    Perform a single equilibrium computation and state update for the system.

    Equilibrium calculations through the `EquilibriumClass`, updates the state based on the computed 
    equilibrium configuration, and visualizes the arm configuration.

    Parameters
    ----------
    Strctr    - StructureClass, Structural definition containing geometry (hinges, edges, node layout, etc.).
    Variabs   - VariablesClass, Material and stiffness parameters for hinges, edges, and stretch elements.
    Sprvsr    - SupervisorClass, Supervisory object controlling simulation and visualization parameters.
    CFG       - user variables in config file
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
    State._save_data(t, Strctr, final_pos, State.buckle_arr, final_F, control_tip_angle=Sprvsr.control_tip_angle)
    print('pos_arr', final_pos)
    print('edge len', Strctr.all_edge_lengths(State.pos_arr))
    print('total edge error', np.sum((Strctr.all_edge_lengths(State.pos_arr)-Strctr.L)**2)/(Strctr.edges*Strctr.L**2))
    plot_funcs.plot_arm(State.pos_arr, State.buckle_arr, State.theta_arr, Strctr.L, modality="measurement")
    plt.show()
    return pos_in_t, final_F


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
    Strctr    - StructureClass, Structural definition containing geometry (hinges, edges, node layout, etc.).
    Variabs   - VariablesClass, Material and stiffness parameters for hinges, edges, and stretch elements.
    Sprvsr    - SupervisorClass, Supervisory object controlling simulation and visualization parameters.
    CFG       - user variables in config file
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

    Notes
    -----
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
        State._save_data(t=i, Strctr=Strctr, pos_arr=final_pos, buckle_arr=State.buckle_arr, Forces=potential_force_in_t,
                         control_tip_angle=Sprvsr.control_tip_angle)
        print("State tip forces ", [State.Fx, State.Fy])
        print("edge lengths ", State.edge_lengths)
        if plot_every > 0 and (i % plot_every == 0):
            plot_funcs.plot_arm(State.pos_arr, State.buckle_arr, State.theta_arr, Strctr.L, modality="measurement")
        Fx_afo_pos[i] = State.Fx
        pos_frames[i, :, :] = State.pos_arr
        buckle_frames[i, :, :] = State.buckle_arr
        theta_frames[i, :] = State.theta_arr

    t1 = time.time()  # stop counting time of computation
    print(f"Total runtime: {t1 - t0:.2f} seconds")  # print time of computation

    return Fx_afo_pos, pos_frames, buckle_frames, theta_frames
