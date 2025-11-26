import numpy as np
import matplotlib.pyplot as plt
import time

from numpy.typing import NDArray
from typing import TYPE_CHECKING, Tuple, Optional


import plot_funcs

from StructureClass import StructureClass
from VariablesClass import VariablesClass
from StateClass import StateClass
from SupervisorClass import SupervisorClass
from EquilibriumClass import EquilibriumClass

if TYPE_CHECKING:
    from StructureClass import StructureClass
    from VariablesClass import VariablesClass
    from StateClass import StateClass
    from SupervisorClass import SupervisorClass
    from EquilibriumClass import EquilibriumClass


def compress_to_tip_pos(Strctr: "StructureClass", Variabs: "VariablesClass", Sprvsr: "SupervisorClass", rand_key: int,
                        tip_pos_i, tip_angle_i, tip_pos_f, tip_angle_f, Eq_iterations, T_eq: int, damping: float, mass: float,
                        tolerance: float, buckle: NDArray) -> Tuple["StateClass", NDArray, NDArray]:
    """
        Incrementally compress origami toward final tip position and angle, ensuring stable convergence, in Eq_iterations steps.

        This function performs a sequence of equilibrium simulations, gradually moving the tip 
        from an initial position and angle `(tip_pos_i, tip_angle_i)` to a final configuration 
        `(tip_pos_f, tip_angle_f)`. After each intermediate step, the new equilibrium state 
        becomes the starting point for the next iteration. This progressive approach stabilizes 
        the numerical integration for stiff systems.

        Parameters
        ----------
        Strctr : StructureClass
            Structural definition containing geometry (hinges, edges, node layout, etc.).
        Variabs : VariablesClass
            Material and stiffness parameters for hinges, edges, and stretch elements.
        Sprvsr : SupervisorClass
            Supervisory object controlling simulation and visualization parameters.
        tip_pos_i : array-like of shape (2,)
            Initial position of the structure’s tip node.
        tip_angle_i : float
            Initial angular orientation of the tip node (radians).
        tip_pos_f : array-like of shape (2,)
            Target final position of the structure’s tip node.
        tip_angle_f : float
            Target final tip orientation (radians).
        Eq_iterations : int
            Number of incremental equilibrium steps between the initial and final configurations.
        T_eq : float
            Total simulation time for each equilibrium calculation.
        damping : float
            Damping coefficient for the dynamic equilibrium solver.
        mass : float
            Mass parameter controlling inertial terms in the equilibrium solver.
        tolerance : float
            Tolerance for adaptive integration accuracy in the dynamics solver.
        buckle : NDArray
            Initial buckle state of all hinges (typically ±1).

        Returns
        -------
        State : StateClass
            Final equilibrium state of the structure after all compression steps.
        pos_in_t : list[NDArray]
            List of position histories over all equilibrium phases.
            Each entry corresponds to the node positions over time during one equilibrium phase.
        force_in_t : list[NDArray]
            List of potential force histories corresponding to each equilibrium phase.

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
        State, pos_in_t_i, force_in_t_0 = one_shot(Strctr, Variabs, Sprvsr, T_eq, damping, mass, tolerance, buckle, tip_pos,
                                                   tip_angle, init_pos=pos_init)
        pos_in_t.append(pos_in_t_i)
        force_in_t.append(force_in_t_0)

    # final one is just some more time
    pos_init = pos_in_t[-1][-1]
    T_eq = 1 * T_eq  # increase time for equilibrium
    damping = 3 * damping  # increase damping
    State, pos_in_t_i, force_in_t_0 = one_shot(Strctr, Variabs, Sprvsr, T_eq, damping, mass, tolerance, rand_key, buckle,
                                               tip_pos, tip_angle, init_pos=pos_init)
    pos_in_t.append(pos_in_t_i)
    force_in_t.append(force_in_t_0)
    return State, pos_in_t, force_in_t


def one_shot(Strctr: "StructureClass", Variabs: "VariablesClass", Sprvsr: "SupervisorClass",
             T_eq: int, damping: float, mass: float, tolerance: float, rand_key: int, buckle: NDArray,
             tip_pos: NDArray, tip_angle: NDArray, init_pos: NDArray = None) -> Tuple["StateClass", NDArray, NDArray]:
    """
    Perform a single equilibrium computation and state update for the system.

    Equilibrium calculations through the `EquilibriumClass`, updates the state based on the computed 
    equilibrium configuration, and visualizes the arm configuration.

    Parameters
    ----------
    Strctr : StructureClass
        Object defining the geometric and topological properties of the structure 
        (e.g., edges, rest lengths, etc.).
    Variabs : VariablesClass
        Object containing simulation parameters such as stiffness coefficients, 
        torque functions, and other physical properties.
    Sprvsr : SupervisorClass
        Supervisor object managing high-level simulation settings and control 
        flags such as tip control behavior.
    T_eq : int
        Total equilibrium time or number of iterations for the equilibrium computation.
    n_steps : int
        Number of discrete steps used in the equilibrium calculation.
    damping : float
        Damping coefficient applied during the dynamic relaxation or equilibrium process.
    mass : float
        Mass parameter affecting the inertia in the dynamic equilibrium simulation.
    calc_through_energy : bool
        Whether to compute equilibrium via energy minimization instead of force balance.
    buckle : NDArray
        Array indicating the initial buckle configuration of the system (typically ±1).
    tip_pos : NDArray
        Prescribed position of the tip node(s) during the equilibrium calculation.
    tip_angle : NDArray
        Prescribed angular orientation of the tip node(s).

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
    State = StateClass(Variabs, Strctr, Sprvsr, buckle_arr=buckle, pos_arr=init_pos)  # buckle defaults to +1

    # --- initialize, no tip movement yet
    Eq = EquilibriumClass(Strctr, T_eq, damping, mass, tolerance, buckle_arr=buckle, pos_arr=State.pos_arr)
    final_pos, pos_in_t, vel_in_t, potential_force_in_t = Eq.calculate_state(Variabs, Strctr, Sprvsr, rand_key,
                                                                             tip_pos=tip_pos, tip_angle=tip_angle)
    State._save_data(0, Strctr, final_pos, State.buckle_arr, potential_force_in_t, compute_thetas_if_missing=True,
                     control_tip_angle=Sprvsr.control_tip_angle)
    print('pos_arr', final_pos)
    print('edge len', Strctr.all_edge_lengths(State.pos_arr))
    print('total edge error', np.sum((Strctr.all_edge_lengths(State.pos_arr)-Strctr.L)**2)/(Strctr.edges*Strctr.L**2))
    # print('stretch energy', State.stretch_energy(Variabs, Strctr))
    # print('torque energy', State.bending_energy(Variabs, Strctr))
    print('Fx', State.Fx)
    plot_funcs.plot_arm(State.pos_arr, State.buckle_arr, State.theta_arr, Strctr.L, modality="measurement")
    plt.show()
    return State, pos_in_t, potential_force_in_t


def ADMET_stress_strain(Strctr: StructureClass, Variabs: VariablesClass, Sprvsr: SupervisorClass, State: StateClass,  T_eq: float,
                        damping: float, mass: float, tolerance: float, tip_angle: float, rand_key, *,
                        pos_noise: Optional[float] = None, vel_noise: Optional[float] = None, plot_every: int = 1
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
    T_eq : float
        Equilibrium simulation time horizon for each step (passed to `EquilibriumClass`).
    damping : float
        Damping coefficient used in the dynamics solver.
    mass : float
        Mass parameter for the dynamics solver.
    tolerance : float
        Tolerance for adaptive integration in the dynamics solver.
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
        if i == 0:
            Eq = EquilibriumClass(Strctr, T_eq, damping, mass, tolerance, buckle_arr=State.buckle_arr)
        else:
            Eq = EquilibriumClass(Strctr, T_eq, damping, mass, tolerance, buckle_arr=State.buckle_arr, pos_arr=final_pos)
        final_pos, pos_in_t, vel_in_t, potential_force_in_t = Eq.calculate_state(Variabs, Strctr, Sprvsr, rand_key,
                                                                                 tip_pos=tip_pos, tip_angle=tip_angle,
                                                                                 pos_noise=pos_noise, vel_noise=vel_noise)

        # Buckle
        State.buckle(Variabs, Strctr, i, State)

        # Save, plot, store
        State._save_data(t=i, Strctr=Strctr, pos_arr=final_pos, buckle_arr=State.buckle_arr, Forces=potential_force_in_t,
                         compute_thetas_if_missing=True, control_tip_angle=Sprvsr.control_tip_angle)
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
