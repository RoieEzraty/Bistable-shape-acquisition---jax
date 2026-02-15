from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp
from pathlib import Path
from scipy.signal import savgol_filter
import csv

from typing import Tuple, List
from numpy import array, zeros
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional, Literal, Dict, Any

import helpers_builders

if TYPE_CHECKING:
    from SupervisorClass import SupervisorClass
    from VariablesClass import VariablesClass
    from EquilibriumClass import EquilibriumClass


# ===================================================
# file_funcs - functions to assist with file conversions etc.
# ===================================================


def export_stress_strain_sim(Sprvsr: "SupervisorClass", Fx_afo_pos: NDArray[np.float_], Fy_afo_pos: NDArray[np.float_], 
                             L: float, buckle_arr: NDArray[np.int], filename: str = None) -> None:

    # --- build pandas dataframe ---
    df = pd.DataFrame({
        "x_tip": Sprvsr.tip_pos_in_t[:, 0],
        "y_tip": Sprvsr.tip_pos_in_t[:, 1],
        "tip_angle_rad": Sprvsr.tip_angle_in_t,
        "Fx": Fx_afo_pos,
        "Fy": Fy_afo_pos,
    })
    if filename is not None:
        pass 
    else:
        filename = f"L={L}_buckle{buckle_arr.reshape(-1)}.csv"  # filename example "L=1_buckle1111.csv"
    out_path = Path(filename)
    df.to_csv(out_path, index=False)


def import_stress_strain_sim_and_plot(path: str, plot: bool = False) -> df:
    sim_df = pd.read_csv(path)   # assumes the header row is in the file
    if plot:
        plt.plot(sim_df['x_tip'], sim_df['Fx'])
        plt.xlabel('tip pos')
        plt.ylabel('Fx')
        plt.show()
    return sim_df


def import_stress_strain_exp_and_plot(path: str, plot: bool = True) -> None:
    exp_df = pd.read_csv(path)   # assumes the header row is in the file
    if plot:
        plt.plot(exp_df['Position (mm)'], exp_df['Load2 (N)'])
        plt.xlabel('tip pos')
        plt.ylabel('Fx')
        plt.show()
    return exp_df


def load_pos_force(path: str, mod: Literal["dict", "arrays"] = "dict", stretch_factor: Optional[float] = None):

    if mod == "dict":
        rows: List[Dict[str, Any]] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append({
                    "t_unix": float(r["t_unix"]),
                    "pos": (
                        float(r["pos_x"]),
                        float(r["pos_y"]),
                        float(r["pos_z"]),
                    ),
                    "force": (
                        float(r["force_x"]),
                        float(r["force_y"]),
                    ),
                })
        return rows

    elif mod == "arrays":
        T, P, F = [], [], []

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for r in reader:
                # ---- time ----
                t_val = helpers_builders._get_first_in_file(r, ["t_unix", "time", "t"], name="time", allow_missing=True)
                if t_val is not None:
                    T.append(t_val)

                # ---- position / tip pose ----
                x = helpers_builders._get_first_in_file(r, ["pos_x", "x_tip", "Px"], name="x")
                y = helpers_builders._get_first_in_file(r, ["pos_y", "y_tip", "Py"], name="y")
                theta = helpers_builders._get_first_in_file(r, ["pos_z", "theta", "tip_angle_rad"], name="theta")

                if stretch_factor is not None:
                    x *= stretch_factor
                    y *= stretch_factor

                P.append([x, y, np.deg2rad(theta)])

                # ---- forces ----
                Fx = helpers_builders._get_first_in_file(r, ["force_x", "F_x", "Fx"], name="Fx")
                Fy = helpers_builders._get_first_in_file(r, ["force_y", "F_y", "Fy"], name="Fy")

                F.append([Fx, Fy])

        return (np.asarray(T, dtype=float), np.asarray(P, dtype=float), np.asarray(F, dtype=float))

    else:
        raise ValueError(f"Unknown mode: {mod}")


def build_torque_and_k_from_file(path: str, *, contact: bool = True, angles_in_degrees: bool = True, savgol_window: int = None,
                                 contact_scale: float = 1e2) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray,
                                                                      callable, callable]:
    """
    Load (angle, torque) from text file and construct JAX-friendly interpolants.
    contact: bool = if True, account for contact between plastic faces where torques explode

    Returns
    -------
    theta_grid : jnp.ndarray, shape (N,)
        Sorted grid of angles (radians).
    torque_grid : jnp.ndarray, shape (N,)
        Torque samples on grid.
    k_grid : jnp.ndarray, shape (N,)
        Stiffness samples (numeric derivative of torque w.r.t. theta).
    torque_of_theta : callable
        JAX function theta -> torque (linear interpolation, clamped at ends).
    k_of_theta : callable
        JAX function theta -> stiffness (linear interpolation, clamped at ends).
    """
    # --- load ---
    try:
        data = np.loadtxt(path)                # shape (N, 2)
    except ValueError:
        data = np.loadtxt(path, delimiter=',')
    theta = data[:, 0]
    tau = data[:, 1]

    if path == "Roie_metal_singleMylar_short.csv":  # flip axes
        tau = -tau

    # degrees -> radians if needed
    if angles_in_degrees:
        theta = np.deg2rad(theta)

    # --- sort & unique (interp requires monotonic x) ---
    order = np.argsort(theta)
    theta = theta[order]
    tau = tau[order]
    # collapse duplicates (if any)
    theta_u, idx = np.unique(theta, return_index=True)
    tau_u = tau[idx]

    # --- numeric derivative: k = d(tau)/d(theta) ---
    # central differences on nonuniform grid
    # (np.gradient handles non-uniform spacing if you pass x)
    k = np.gradient(tau_u, theta_u)

    # optional light smoothing of k (pure NumPy, outside JAX)
    if savgol_window is not None and savgol_window > 2 and savgol_window % 2 == 1:
        try:
            k = savgol_filter(k, window_length=savgol_window, polyorder=4, mode="interp")
        except Exception:
            print('SciPy isnt available, just skip smoothing')

    # --- wrap as JAX arrays ---
    theta_grid = jnp.asarray(theta_u, dtype=jnp.float32)
    torque_grid = jnp.asarray(tau_u, dtype=jnp.float32)
    k_grid = jnp.asarray(k, dtype=jnp.float32)
    k_grid = k_grid.at[k_grid < 0].set(10e-4)  # for numerical stability, singular point of experimental negative k

    # --- clamped linear interpolators (JAX) ---
    def _clamp(x, xmin, xmax):
        return jnp.clip(x, xmin, xmax)

    def torque_of_theta(theta_query: jnp.ndarray) -> jnp.ndarray:
        # masks for outside vs inside range
        above = theta_query > theta_grid[-1]
        below = theta_query < theta_grid[0]
        th = _clamp(theta_query, theta_grid[0], theta_grid[-1])
        tau = jnp.interp(th, theta_grid, torque_grid)  # torque
        if contact:  # account for plates in contact, torque diverges
            # masks for outside vs inside range
            above = theta_query > theta_grid[-1]
            below = theta_query < theta_grid[0]
            above_parabola = contact_scale * jnp.max(k_grid) * (theta_query - theta_grid[-1])**2 + jnp.max(torque_grid)
            below_parabola = - contact_scale * jnp.max(k_grid) * (theta_query - theta_grid[0])**2 + jnp.min(torque_grid)

            # tau = jnp.where(above, contact_scale * jnp.max(torque_grid), tau)
            # tau = jnp.where(below, contact_scale * jnp.min(torque_grid), tau)
            tau = jnp.where(above, above_parabola, tau)
            tau = jnp.where(below, below_parabola, tau)
        return tau

    def k_of_theta(theta_query: jnp.ndarray) -> jnp.ndarray:
        th = _clamp(theta_query, theta_grid[0], theta_grid[-1])
        return jnp.interp(th, theta_grid, k_grid)

    return theta_grid, torque_grid, k_grid, torque_of_theta, k_of_theta


def export_training_csv(path_csv: str, Strctr, Sprvsr, T=None, State_meas=None, State_update=None, State_des=None):
    """
    Save one row per training step t.
    """
    path_csv = Path(path_csv)
    path_csv.parent.mkdir(parents=True, exist_ok=True)

    if T is None:
        T = int(Sprvsr.T)
    H = int(Strctr.hinges)
    S = int(Strctr.shims)

    has_angle = (getattr(Sprvsr, "tip_angle_in_t", None) is not None) and (Sprvsr.tip_angle_in_t is not None)
    has_upd_angle = (getattr(Sprvsr, "tip_angle_update_in_t", None) is not None) and (Sprvsr.tip_angle_update_in_t is not None)

    # ---- header ----
    header = ["t",
              "tip_x", "tip_y"]
    if has_angle:
        header += ["tip_angle"]

    header += ["upd_tip_x", "upd_tip_y"]
    if has_upd_angle:
        header += ["upd_tip_angle"]

    # loss columns (Sprvsr.loss_in_t is (T, loss_size))
    loss_size = Sprvsr.loss_in_t.shape[1]
    header += [f"loss_{i}" for i in range(loss_size)]
    header += ["loss_MSE"]

    # measured
    if State_meas is not None:
        header += ["Fx_meas", "Fy_meas"]

    # desired
    header += ["Fx_des", "Fy_des"]

    # buckle (from update state ideally)
    if State_update is not None:
        for h in range(H):
            for s in range(S):
                header.append(f"buckle_h{h}_s{s}")

    # ---- write ----
    with open(path_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        for t in range(T):
            row = [t,
                   float(Sprvsr.tip_pos_in_t[t, 0]), float(Sprvsr.tip_pos_in_t[t, 1])]
            if has_angle:
                row += [float(Sprvsr.tip_angle_in_t[t])]

            row += [float(Sprvsr.tip_pos_update_in_t[t, 0]), float(Sprvsr.tip_pos_update_in_t[t, 1])]
            if has_upd_angle:
                row += [float(Sprvsr.tip_angle_update_in_t[t])]

            row += [float(x) for x in Sprvsr.loss_in_t[t, :]]

            row += [float(Sprvsr.loss_MSE_in_t[t])]

            if State_meas is not None:
                row += [float(State_meas.Fx_in_t[t]),
                        float(State_meas.Fy_in_t[t])]

            row += [float(Sprvsr.desired_Fx_in_t[t]),
                    float(Sprvsr.desired_Fy_in_t[t])]

            if State_update is not None:
                B = State_update.buckle_in_t[:, :, t]  # (H,S)
                row += [int(B[h, s]) for h in range(H) for s in range(S)]

            w.writerow(row)


def export_training_npz(path_npz: str, **arrays):
    """
    Save big arrays (pos/angles/buckles) in one compressed file.
    """
    path_npz = Path(path_npz)
    path_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path_npz, **arrays)
