from __future__ import annotations

import csv
import copy
import re
import numpy as np
import json
import pandas as pd
import jax.numpy as jnp
from pathlib import Path
from scipy.signal import savgol_filter
from collections import deque

from typing import Tuple, List
from numpy import array, zeros
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional, Literal, Dict, Any

import helpers_builders

if TYPE_CHECKING:
    from SupervisorClass import SupervisorClass
    from StateClass import StateClass
    from StructureClass import StructureClass


# ===================================================
# file_funcs - functions to assist with file conversions etc.
# ===================================================

# ---------------------------------------------------------------
# Imports
# ---------------------------------------------------------------
def load_pos_force(path: str, mod: Literal["dict", "arrays"] = "dict",
                   stretch_factor: Optional[float] = None) -> Union[List[Dict[str, Any]],
                                                                    Tuple[NDArray[np.float64], NDArray[np.float64],
                                                                          NDArray[np.float64]]]:
    """
    Load tip positions and forces from a CSV file using csv.DictReader, and convert it into either:
    - a list of dictionaries (`mod="dict"`)
    - NumPy arrays (`mod="arrays"`)

    Parameters
    ----------
    path           : str, Path to the CSV file.
    mod            : {"dict", "arrays"}, default="dict"
                     - `"dict"`   = list of dictionaries with keys `"t_unix"`, `"pos"`, `"force"`
                     - `"arrays"` = tuple `(T, P, F)` of NumPy arrays
    stretch_factor : Optional[float], Optional inverse scaling applied to x and y positions.
                                      For training CSVs exported with ``Sprvsr.convert_pos=1000``,
                                      pass ``stretch_factor=1000`` to convert mm back to m.

    Returns
    -------
    rows : list[dict]
          - `"t_unix"` : float
          - `"pos"`    : tuple (x, y, tip_angle_deg)
          - `"force"`  : tuple (Fx, Fy)

          OR
          (T, P, F) : tuple of ndarrays
          T : ndarray, shape (N,) Time values (may be empty if the file contains no time column).
          P : ndarray, shape (N, 3), Tip pose values `[x, y, theta [rad]]`.
          F : ndarray, shape (N, 2), Tip force values `[Fx, Fy]`.

    Notes
    -----
    - The loader accepts multiple possible column names for compatibility
      with different datasets (e.g. `"x_tip"`, `"pos_x"`, `"Px"`).
    - Angles always returned in **radians** when `mod="arrays"`.
    """
    if mod == "dict":
        rows: List[Dict[str, Any]] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append({
                    "t_unix": float(r["t_unix"]),
                    "pos": (
                        float(r["x_tip"]),
                        float(r["y_tip"]),
                        float(r["tip_angle_deg"]),
                    ),
                    "force": (
                        float(r["F_x"]),
                        float(r["F_y"]),
                    ),
                })
        return rows

    elif mod == "arrays":
        T, P, F = [], [], []

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for r in reader:
                # ---- time ----
                t_val, _ = helpers_builders._get_first_in_file(r, ["t_unix", "time", "t"], name="time",
                                                               allow_missing=True)
                if t_val is not None:
                    T.append(t_val)

                # ---- position / tip pose ----
                x, _ = helpers_builders._get_first_in_file(r, ["pos_x", "x_tip", "Px"], name="x")
                y, _ = helpers_builders._get_first_in_file(r, ["pos_y", "y_tip", "Py"], name="y")
                theta, theta_key = helpers_builders._get_first_in_file(r, ["theta", "tip_angle_rad",
                                                                           "tip_angle_deg", "pos_z"], name="theta")

                if stretch_factor is not None:
                    x *= stretch_factor
                    y *= stretch_factor
                if theta_key != "tip_angle_rad":
                    theta = np.deg2rad(theta)

                P.append([x, y, theta])

                # ---- forces ----
                Fx, _ = helpers_builders._get_first_in_file(r, ["force_x", "F_x", "Fx"], name="Fx")
                Fy, _ = helpers_builders._get_first_in_file(r, ["force_y", "F_y", "Fy"], name="Fy")

                F.append([Fx, Fy])

        return (np.asarray(T, dtype=float), np.asarray(P, dtype=float), np.asarray(F, dtype=float))

    else:
        raise ValueError(f"Unknown mode: {mod}")


def load_full_pos_in_t(path: str | Path, stretch_factor: Optional[float] = None) -> NDArray[np.float64]:
    """Load full arm positions in time from a CSV column.

    Parameters
    ----------
    path : str | Path
        CSV path containing a ``final_pos_update`` column.
    stretch_factor : float | None, optional
        If provided, multiply all positions by this factor.

    Returns
    -------
    NDArray[np.float64]
        Positions in time, shape ``(T, N, 2)``.
    """
    P_lst: list[NDArray[np.float64]] = []
    B_lst: list[NDArray[np.float64]] = []

    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for r in reader:
            pos, _ = helpers_builders._get_first_in_file(r, ["final_pos_update"], name="final_pos_update",
                                                         type="NDArray")
            buckle, _ = helpers_builders._get_first_in_file(r, ["buckle_arr_update"], name="buckle_arr_update",
                                                            type="NDArray")

            pos = np.asarray(pos, dtype=float)
            buckle = np.asarray(buckle, dtype=float)

            if stretch_factor is not None:
                pos = pos * stretch_factor

            P_lst.append(pos)
            B_lst.append(buckle)

    if not P_lst:
        return np.empty((0, 0, 2), dtype=float)
    if not B_lst:
        return np.empty((0, 0), dtype=float)

    return np.stack(P_lst, axis=0), np.stack(B_lst, axis=0)


def load_training(path: str, stretch_factor: Optional[float] = None, *,
                  include_desired_pose: bool = False,
                  norm_angle: float = np.pi) -> tuple[NDArray[np.float64], ...]:
    """
    Load tip positions and forces from a CSV file using csv.DictReader, and convert it into either:
    - a list of dictionaries (`mod="dict"`)
    - NumPy arrays (`mod="arrays"`)

    Parameters
    ----------
    path           : str, Path to the CSV file.
    stretch_factor : Optional[float], Optional scaling applied to x and y positions,
                                      for rescaling experimental trajectories.
    include_desired_pose : bool, optional
        If True, also return desired tip positions and angles. For legacy
        position-training files without desired-pose columns, reconstruct them
        from the stored normalized loss.
    norm_angle : float, optional
        Angle normalization used to reconstruct legacy desired angles.

    Returns
    -------

    L        - ndarray, shape (T, loss_dim), loss_0/loss_1/(optional loss_2...)
    B        - ndarray, shape (T, H, S), measured buckle arrays
    P_update - ndarray, shape (T, 2), updated tip positions
    A_update - ndarray, shape (T,), updated tip angles in radians
    tip_des  - ndarray, shape (T, 2), desired tip positions; returned only when
               ``include_desired_pose=True``
    angle_des - ndarray, shape (T,), desired tip angles in radians; returned only
                when ``include_desired_pose=True``

    Notes
    -----
    - The loader accepts multiple possible column names for compatibility
      with different datasets (e.g. `"x_tip"`, `"pos_x"`, `"Px"`).
    - Angles always returned in **radians** when `mod="arrays"`.
    """
    L, B = [], []
    P_meas, P_update = [], []
    tip_update, angle_update = [], []
    tip_des, angle_des = [], []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        loss_cols = [c for c in (reader.fieldnames or []) if re.fullmatch(r"loss_\d+", c)]
        loss_cols = sorted(loss_cols, key=lambda c: int(c.split("_")[1]))

        if len(loss_cols) == 0:
            raise KeyError(f"No loss_* columns found in {path}")

        for r in reader:
            # ------ Loss: infer dim from existing loss_* columns ------
            loss_row = []
            for c in loss_cols:
                val, _ = helpers_builders._get_first_in_file(r, [c], name=c)
                loss_row.append(val)
            L.append(loss_row)

            # ------ Buckle ------
            Buckle, _ = helpers_builders._get_first_in_file(r, ["buckle_arr_meas", "buckle_arr_update"],
                                                            name="buckle_arr",
                                                            type="NDArray")
            if Buckle.ndim == 1:
                Buckle = Buckle.reshape(-1, 1)
            B.append(Buckle)

            # ---- tip position / angle ----
            X_update, _ = helpers_builders._get_first_in_file(r, ["upd_x_tip"], name="upd_x_tip")
            Y_update, _ = helpers_builders._get_first_in_file(r, ["upd_y_tip"], name="upd_y_tip")
            Angle_update, theta_key = helpers_builders._get_first_in_file(r, ["upd_tip_angle"],
                                                                          name="upd_tip_angle")

            if stretch_factor is not None:
                X_update /= stretch_factor
                Y_update /= stretch_factor

            # export_training_csv writes upd_tip_angle in degrees, so convert to radians
            if theta_key != "tip_angle_rad":
                Angle_update = np.deg2rad(Angle_update)

            tip_update.append([X_update, Y_update])
            angle_update.append(Angle_update)

            # ------ Desired tip pose ------
            if include_desired_pose:
                X_des, _ = helpers_builders._get_first_in_file(
                    r, ["x_des_tip"], name="x_des_tip", allow_missing=True
                )
                Y_des, _ = helpers_builders._get_first_in_file(
                    r, ["y_des_tip"], name="y_des_tip", allow_missing=True
                )
                Angle_des, angle_des_key = helpers_builders._get_first_in_file(
                    r, ["des_tip_angle"], name="des_tip_angle", allow_missing=True
                )
                if X_des is not None and Y_des is not None and Angle_des is not None:
                    if stretch_factor is not None:
                        X_des /= stretch_factor
                        Y_des /= stretch_factor
                    if angle_des_key != "tip_angle_rad":
                        Angle_des = np.deg2rad(Angle_des)
                    tip_des.append([X_des, Y_des])
                    angle_des.append(Angle_des)
                else:
                    tip_des.append(None)
                    angle_des.append(None)

            # ------ Full positions ------
            pos_meas, _ = helpers_builders._get_first_in_file(r, ["final_pos_meas"], name="final_pos_meas",
                                                              type="NDArray", allow_missing=True)
            pos_update, _ = helpers_builders._get_first_in_file(r, ["final_pos_update"], name="final_pos_update",
                                                                type="NDArray", allow_missing=True)

            if pos_meas is not None:
                pos_meas = np.asarray(pos_meas, dtype=float)
                if pos_meas.ndim == 1:
                    pos_meas = pos_meas.reshape(-1, 2)
                if stretch_factor is not None:
                    pos_meas /= stretch_factor
                P_meas.append(pos_meas)

            if pos_update is not None:
                pos_update = np.asarray(pos_update, dtype=float)
                if pos_update.ndim == 1:
                    pos_update = pos_update.reshape(-1, 2)
                if stretch_factor is not None:
                    pos_update /= stretch_factor
                P_update.append(pos_update)

    L = np.asarray(L, dtype=float)

    B = np.stack(B, axis=0)          # (T, H, 1)

    P_meas = np.asarray(P_meas, dtype=float)
    P_update = np.asarray(P_update, dtype=float)

    tip_update = np.asarray(tip_update, dtype=float)
    angle_update = np.asarray(angle_update, dtype=float)

    result = (L, B, P_meas, P_update, tip_update, angle_update)
    if not include_desired_pose:
        return result

    missing_desired = any(pos is None for pos in tip_des)
    if missing_desired:
        if P_meas.shape[0] != L.shape[0] or L.shape[1] < 3:
            raise KeyError(
                "Desired tip-pose columns are missing and cannot be reconstructed "
                "without measured geometry and three position-loss components"
            )
        edge_lengths = np.linalg.norm(np.diff(P_meas, axis=1), axis=2)
        norm_pos = float(np.mean(edge_lengths[edge_lengths > 0]))
        measured_tip = P_meas[:, -1, :]
        measured_segments = P_meas[:, -1, :] - P_meas[:, -2, :]
        measured_angle = np.arctan2(measured_segments[:, 1], measured_segments[:, 0])
        tip_des_arr = measured_tip + L[:, :2] * norm_pos
        angle_des_arr = measured_angle + L[:, 2] * norm_angle
    else:
        tip_des_arr = np.asarray(tip_des, dtype=float)
        angle_des_arr = np.asarray(angle_des, dtype=float)

    return result + (tip_des_arr, angle_des_arr)


def warm_start_training_from_csv(path: str, Sprvsr: "SupervisorClass", State_meas: "StateClass",
                                 State_update: "StateClass", Strctr: "StructureClass", t_update: int,
                                 stretch_factor: Optional[float] = None,
                                 State_des: Optional["StateClass"] = None) -> int:
    """Load CSV rows with ``t < t_update`` into existing training objects."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if int(r["t"]) < t_update]
    if not rows:
        raise ValueError(f"No rows with t < {t_update} found in {path}")
    if t_update > Sprvsr.T:
        raise ValueError(f"t_update={t_update} exceeds Sprvsr.T={Sprvsr.T}")

    L, B, P_meas, P_update, tip_update, angle_update = load_training(path, stretch_factor=stretch_factor)
    loss_cols = sorted([c for c in rows[0] if re.fullmatch(r"loss_\d+", c)], key=lambda c: int(c.split("_")[1]))
    for i, r in enumerate(rows):
        t = int(r["t"])
        Sprvsr.tip_pos_update_in_t[t, :] = tip_update[i]
        Sprvsr.tip_angle_update_in_t[t] = angle_update[i]
        Sprvsr.total_angle_update_in_t[t] = helpers_builders._get_total_angle(Sprvsr.tip_pos_update_in_t[t, :],
                                                                               0.0 if t == 0 else Sprvsr.total_angle_update_in_t[t - 1],
                                                                               Strctr.L)
        Sprvsr.loss_in_t[t, :len(loss_cols)] = L[i, :len(loss_cols)]
        Sprvsr.loss_MSE_in_t[t] = float(r.get("loss_MSE", 0.0))
        State_meas.pos_arr_in_t[:, :, t] = P_meas[i]
        State_update.pos_arr_in_t[:, :, t] = P_update[i]
        State_meas.buckle_in_t[:, :, t] = B[i]
        State_update.buckle_in_t[:, :, t] = helpers_builders._get_first_in_file(
            r, ["buckle_arr_update"], name="buckle_arr_update", type="NDArray")[0].reshape(Strctr.hinges, Strctr.shims)
        State_meas.Fx_in_t[t], State_meas.Fy_in_t[t] = float(r.get("Fx_meas", 0.0)), float(r.get("Fy_meas", 0.0))
        State_update.Fx_in_t[t], State_update.Fy_in_t[t] = float(r.get("Fx_update", 0.0)), float(r.get("Fy_update", 0.0))
        if State_des is not None:
            State_des.Fx_in_t[t], State_des.Fy_in_t[t] = float(r.get("Fx_des", 0.0)), float(r.get("Fy_des", 0.0))

    last_t = int(rows[-1]["t"])
    Sprvsr.loss = Sprvsr.loss_in_t[last_t, :len(loss_cols)].copy()
    for State in (State_meas, State_update):
        State.pos_arr = State.pos_arr_in_t[:, :, last_t].copy()
        State.buckle_arr = State.buckle_in_t[:, :, last_t].copy()
        State.Fx, State.Fy = State.Fx_in_t[last_t], State.Fy_in_t[last_t]
        State.theta_arr = Strctr.all_hinge_angles(State.pos_arr)
    return last_t


# ---------------------------------------------------------------
# Exports
# ---------------------------------------------------------------
def export_predetermined(Sprvsr: "SupervisorClass", State: "StateClass", filename: Optional[str] = None,
                         order: Optional[str] = 'fwd', stretch_factor: Optional[float] = None) -> None:
    """
    Export a predetermined trajectory and its simulated forces to a CSV file.

    Parameters
    ----------
    Sprvsr   - SupervisorClass, for `tip_pos_in_t`, `tip_angle_in_t` and unit conversion factors.
    State    - StateClass, for force histories `Fx_in_t`, `Fy_in_t`, and final buckle configuration.
    filename - Optional[str], output CSV filename. If None, name is generated automatically from buckle configuration.

    Notes
    -----
    - If `filename` not provided, default filename based on buckle configuration in State.buckle_arr (buckle = -1 → 0)
    - Only first `T` entries of `State.Fx_in_t` and `State.Fy_in_t` exported, where `T = len(Sprvsr.tip_pos_in_t)`.
    """
    if order == 'fwd':
        # ------ init and scale sizes
        T = Sprvsr.tip_pos_in_t.shape[0]

        # convert to [mN] and [deg]
        tip_pos_in_t = Sprvsr.tip_pos_in_t * Sprvsr.convert_pos
        tip_angle_in_t = Sprvsr.tip_angle_in_t * Sprvsr.convert_angle
        Fx_afo_pos = State.Fx_in_t[:T] * Sprvsr.convert_F
        Fy_afo_pos = State.Fy_in_t[:T] * Sprvsr.convert_F

    elif order == 'fwd_and_bcwrd':
        # ------ init and scale sizes
        T = 2*Sprvsr.tip_pos_in_t.shape[0]

        # convert to [mN] and [deg]
        tip_pos_in_t = np.append(Sprvsr.tip_pos_in_t, Sprvsr.tip_pos_in_t[::-1, :], axis=0) * Sprvsr.convert_pos
        tip_angle_in_t = np.append(Sprvsr.tip_angle_in_t, Sprvsr.tip_angle_in_t[::-1]) * Sprvsr.convert_angle
        Fx_afo_pos = State.Fx_in_t[:T] * Sprvsr.convert_F
        Fy_afo_pos = State.Fy_in_t[:T] * Sprvsr.convert_F

    # -------- convert positions from [m] to [mm] or vice verse ------
    if stretch_factor is not None:
        tip_pos_in_t = tip_pos_in_t * stretch_factor

    # ------ pandas dataframe ------
    df = pd.DataFrame({"x_tip": tip_pos_in_t[:, 0], "y_tip": tip_pos_in_t[:, 1], "tip_angle_deg": tip_angle_in_t,
                       "F_x": Fx_afo_pos, "F_y": Fy_afo_pos})

    # ------ filename ------
    if filename is not None:
        pass
    else:
        buckle_str = correct_buckle_string(State.buckle_arr)
        filename = f"buckle={buckle_str}.csv"  # filename example "buckle=0001.csv"
    out_path = Path(filename)

    # ------ save ------
    df.to_csv(out_path, index=False)


def export_training_csv(path_csv: str, Sprvsr: "SupervisorClass", T: Optional[int] = None,
                        State_meas: Optional["StateClass"] = None, State_update: Optional["StateClass"] = None) -> None:
    """
    Export training outputs to a CSV file.

    Parameters
    ----------
    path_csv : str, output CSV file path.
    Sprvsr : SupervisorClass, for Supervisor training data and unit conversion factors
    T : Optional[int], number of training steps to export. If None, full training `Sprvsr.T` is used.
    State_meas : Optional[StateClass], for `Fx_in_t`, `Fy_in_t`. If provided, they are exported as `Fx_meas`, `Fy_meas`.
    State_update : Optional[StateClass], for buckle history (`buckle_in_t`).

    Notes
    -----
    - Each row corresponds to a single training step `t`.
    - Arrays are stored as JSON strings inside single CSV cells.
    - `final_pos_meas` / `final_pos_update` are arrays of shape (nodes, 2).
    - `buckle_arr` is stored as shape (H, S), usually (H, 1).
    """
    def arr_to_json(arr: np.ndarray) -> str:
        """Serialize numpy array as compact JSON string for one CSV cell."""
        return json.dumps(np.asarray(arr).tolist(), separators=(",", ":"))

    # ------ convert scalar channels ------
    tip_pos_update_in_t = Sprvsr.tip_pos_update_in_t * Sprvsr.convert_pos
    angle_update_in_t = Sprvsr.tip_angle_update_in_t * Sprvsr.convert_angle
    des_Fx = Sprvsr.desired_Fx_in_t * Sprvsr.convert_F
    des_Fy = Sprvsr.desired_Fy_in_t * Sprvsr.convert_F
    desired_pos_in_t = Sprvsr.desired_pos_in_t * Sprvsr.convert_pos
    training_mode = "pos" if getattr(Sprvsr, "update_scheme", None) == "pos" else "force"

    meas_Fx = meas_Fy = None
    update_Fx = update_Fy = None
    if State_meas is not None:
        meas_Fx = State_meas.Fx_in_t * Sprvsr.convert_F
        meas_Fy = State_meas.Fy_in_t * Sprvsr.convert_F
    if State_update is not None:
        update_Fx = State_update.Fx_in_t * Sprvsr.convert_F
        update_Fy = State_update.Fy_in_t * Sprvsr.convert_F

    path_csv = Path(path_csv)
    path_csv.parent.mkdir(parents=True, exist_ok=True)

    if T is None:
        T = int(Sprvsr.T)

    # ------ headers ------
    header = ["t", "upd_x_tip", "upd_y_tip", "upd_tip_angle"]

    # final update geometry only; measured geometry is kept out of this compact export.
    if State_update is not None:
        header += ["final_pos_update"]

    # losses
    loss_size = Sprvsr.loss_in_t.shape[1]
    header += [f"loss_{i}" for i in range(loss_size)]
    header += ["loss_MSE"]
    header += ["Hamming_distance"]

    if training_mode == "pos":
        if State_meas is not None:
            header += ["final_pos_meas"]
        header += ["final_pos_des"]

    if training_mode == "force":
        if State_meas is not None:
            header += ["Fx_meas", "Fy_meas"]
        header += ["Fx_des", "Fy_des"]
    else:
        if State_meas is not None:
            header += ["x_meas_tip", "y_meas_tip", "meas_tip_angle"]
        header += ["x_des_tip", "y_des_tip", "des_tip_angle"]

    if State_update is not None:
        header += ["Fx_update", "Fy_update"]

    # update buckle arrays only
    if State_update is not None:
        header += ["buckle_arr_update"]

    # chain intersects with itself
    if State_update is not None and State_update.intersection_times is not None:
        header += ["intersection_times"]

    # ------ write ------
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        for t in range(T):
            row = [
                t,
                float(tip_pos_update_in_t[t, 0]),
                float(tip_pos_update_in_t[t, 1]),
                float(angle_update_in_t[t]),
            ]

            # full updated state positions
            if State_update is not None:
                pos_update = State_update.pos_arr_in_t[:, :, t] * Sprvsr.convert_pos
                row += [arr_to_json(pos_update)]

            # losses
            row += [float(x) for x in Sprvsr.loss_in_t[t, :]]
            row += [float(Sprvsr.loss_MSE_in_t[t])]
            row += [int(Sprvsr.Hamming_distance_in_t[t])]

            if training_mode == "pos":
                if State_meas is not None:
                    pos_meas_full = State_meas.pos_arr_in_t[:, :, t] * Sprvsr.convert_pos
                    row += [arr_to_json(pos_meas_full)]
                row += [arr_to_json(desired_pos_in_t[:, :, t])]

            if training_mode == "force":
                if State_meas is not None:
                    row += [float(meas_Fx[t]), float(meas_Fy[t])]
                row += [float(des_Fx[t]), float(des_Fy[t])]
            else:
                if State_meas is not None:
                    pos_meas = State_meas.pos_arr_in_t[:, :, t]
                    row += [
                        float(pos_meas[-1, 0] * Sprvsr.convert_pos),
                        float(pos_meas[-1, 1] * Sprvsr.convert_pos),
                        float(helpers_builders._get_tip_angle(pos_meas) * Sprvsr.convert_angle),
                    ]
                pos_des = Sprvsr.desired_pos_in_t[:, :, t]
                row += [
                    float(desired_pos_in_t[-1, 0, t]),
                    float(desired_pos_in_t[-1, 1, t]),
                    float(helpers_builders._get_tip_angle(pos_des) * Sprvsr.convert_angle),
                ]

            if State_update is not None:
                row += [float(update_Fx[t]), float(update_Fy[t])]

            # update buckle arrays only
            if State_update is not None:
                row += [arr_to_json(State_update.buckle_in_t[:, :, t])]

            # chain intersects with itself
            if State_update is not None and State_update.intersection_times is not None:
                row += [int(State_update.intersection_times[t]) if t < len(State_update.intersection_times) else int(0)]

            w.writerow(row)


def export_training_npz(path_npz: str, **arrays):
    """
    Save big arrays (pos/angles/buckles) in one compressed file.
    """
    path_npz = Path(path_npz)
    path_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path_npz, **arrays)


def export_tip_grid_buckle_map_npz(path_npz: str | Path, Sprvsr: "SupervisorClass",
                                   buckle_grid_frames: NDArray, *, y_num: Optional[int] = None,
                                   theta_num: Optional[int] = None,
                                   r_num: Optional[int] = None,
                                   phi_num: Optional[int] = None, snake: bool = True,
                                   init_buckle: Optional[NDArray] = None) -> Path:
    """Export final and first-transition buckle states on a tip sweep grid.

    For polar arc sweeps, ``first_buckle_ids`` records the first buckle change
    after arrival at the spatial grid point. Fractional travel is excluded.
    """
    path_npz = Path(path_npz)
    path_npz.parent.mkdir(parents=True, exist_ok=True)

    target_indices = np.asarray(Sprvsr.tip_grid_target_indices, dtype=int)
    start_indices = np.asarray(Sprvsr.tip_grid_target_start_indices, dtype=int)
    frames = np.asarray(buckle_grid_frames, dtype=np.int32)
    target_buckles = frames[target_indices]
    target_points = np.column_stack((
        np.asarray(Sprvsr.tip_grid_base_pos_in_t, dtype=np.float32),
        np.asarray(Sprvsr.tip_grid_base_angle_in_t, dtype=np.float32),
    ))

    is_polar_grid = r_num is not None
    first_num = int(phi_num if is_polar_grid else y_num)
    second_num = int(r_num if is_polar_grid else theta_num)
    polar_indices = getattr(
        Sprvsr, "tip_grid_completed_point_indices", Sprvsr.tip_grid_point_indices
    )
    expected_targets = (
        len(polar_indices) if is_polar_grid
        else 1 + first_num * second_num
    )
    if (len(start_indices) != expected_targets
            or len(target_indices) != expected_targets
            or len(target_points) != expected_targets):
        raise ValueError(
            f"Expected {expected_targets} targets, "
            f"got {len(start_indices)} starts, {len(target_indices)} buckle targets, "
            f"and {len(target_points)} grid points."
        )

    first_buckle_ids = None
    if init_buckle is not None:
        init_buckle_arr = np.asarray(init_buckle, dtype=np.int32).reshape(frames.shape[1:])
        first_ids = np.full(len(target_indices), -1, dtype=np.int32)
        fractions = (
            np.asarray(Sprvsr.tip_grid_path_fraction_in_t, dtype=float)
            if is_polar_grid else None
        )
        for target_i, (start, final) in enumerate(zip(start_indices, target_indices)):
            baseline = init_buckle_arr
            first_frame = int(start)
            if is_polar_grid:
                arrival_offsets = np.flatnonzero(
                    np.isclose(fractions[start:final + 1], 1.0)
                )
                if not len(arrival_offsets):
                    raise ValueError(
                        f"Target {target_i} has no path_fraction=1 arrival frame."
                    )
                arrival_frame = int(start + arrival_offsets[0])
                baseline = frames[arrival_frame]
                first_frame = arrival_frame + 1

            for buckle in frames[first_frame:final + 1]:
                if not np.array_equal(buckle, baseline):
                    first_ids[target_i] = helpers_builders.buckle_to_index(buckle.reshape(-1))
                    break

    if is_polar_grid:
        buckle_matrix = np.zeros(
            (first_num, second_num, *target_buckles.shape[1:]), dtype=np.int32
        )
        grid_points = np.full((first_num, second_num, 3), np.nan, dtype=np.float32)
        buckle_ids = np.zeros((first_num, second_num), dtype=np.int32)
        valid_mask = np.zeros((first_num, second_num), dtype=bool)
        first_buckle_ids = np.full((first_num, second_num), -1, dtype=np.int32)
        for target_i, (phi_idx, r_idx) in enumerate(polar_indices):
            buckle = target_buckles[target_i]
            buckle_matrix[phi_idx, r_idx] = buckle
            grid_points[phi_idx, r_idx] = target_points[target_i]
            buckle_ids[phi_idx, r_idx] = helpers_builders.buckle_to_index(buckle.reshape(-1))
            if init_buckle is not None:
                first_buckle_ids[phi_idx, r_idx] = first_ids[target_i]
            valid_mask[phi_idx, r_idx] = True
    else:
        buckle_matrix = target_buckles[1:].reshape(
            first_num, second_num, *target_buckles.shape[1:]
        )
        grid_points = target_points[1:].reshape(first_num, second_num, 3)
        if snake:
            buckle_matrix[1::2] = buckle_matrix[1::2, ::-1].copy()
            grid_points[1::2] = grid_points[1::2, ::-1].copy()
        buckle_ids = np.asarray([
            helpers_builders.buckle_to_index(buckle.reshape(-1))
            for buckle in buckle_matrix.reshape(-1, *buckle_matrix.shape[2:])
        ], dtype=np.int32).reshape(first_num, second_num)
        if init_buckle is not None:
            first_buckle_ids = first_ids[1:].reshape(first_num, second_num)
            if snake:
                first_buckle_ids[1::2] = first_buckle_ids[1::2, ::-1].copy()

    arrays: dict[str, Any] = {
        "buckle_matrix": buckle_matrix,
        "buckle_ids": buckle_ids,
        "grid_points": grid_points,
        "y_values": grid_points[:, 0, 1],
        "theta_values": grid_points[0, :, 2],
        "x_values": grid_points[:, :, 0] if is_polar_grid else grid_points[:, 0, 0],
        "grid_axes": np.asarray("polar_xy" if is_polar_grid else "ytheta"),
        "source_snake": np.bool_(snake),
        "y_scale": np.float32(Sprvsr.convert_pos),
    }
    if is_polar_grid:
        arrays["valid_mask"] = valid_mask
        arrays["r_values"] = np.asarray(Sprvsr.tip_grid_r_values)
        arrays["phi_values"] = np.asarray(Sprvsr.tip_grid_phi_values)
        arrays["r_num"] = np.int32(second_num)
        arrays["phi_num"] = np.int32(first_num)
    else:
        arrays["y_num"] = np.int32(first_num)
        arrays["theta_num"] = np.int32(second_num)
    if init_buckle is not None:
        arrays["init_buckle"] = np.asarray(init_buckle, dtype=np.int32)
        arrays["first_buckle_ids"] = first_buckle_ids
    if hasattr(Sprvsr, "tip_grid_angle_sequences"):
        arrays["angle_sequences"] = np.asarray(
            Sprvsr.tip_grid_angle_sequences, dtype=np.float32
        )
    if hasattr(Sprvsr, "tip_grid_transition_csv_mode"):
        arrays["transition_csv_mode"] = np.asarray(
            Sprvsr.tip_grid_transition_csv_mode
        )

    np.savez_compressed(path_npz, **arrays)
    return path_npz


def load_tip_grid_buckle_maps_from_csvs(
        directory: str | Path, *, pos_scale: float = 1000.0,
        angle_scale: float = 180.0 / np.pi
        ) -> dict[str, dict[str, Any]]:
    """Reconstruct canonical tip-grid buckle maps from transition CSV files.

    Parameters
    ----------
    directory : str or Path
        Directory containing files named
        ``init_<bits>_finalTip_x=<x>_y=<y>_theta=<theta>.csv``.
    pos_scale : float, default=1000
        Position export scale used by the CSVs. Exported positions are divided
        by this value so returned grid points are in simulation units.
    angle_scale : float, default=180/pi
        Angle export scale used by the CSVs. Exported angles are divided by
        this value so returned grid points are in radians.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping from initial buckle bit string to a grid-data dictionary. Each
        dictionary contains:

        - ``buckle_matrix``: ``(y, theta, hinges, shims)``
        - ``buckle_grid_frames``: flattened canonical grid frames
        - ``buckle_ids``: ``(y, theta)``
        - ``grid_points``: ``(y, theta, 3)`` storing ``[x, y, theta]``
        - ``x_values``, ``y_values``, and ``theta_values``
        - ``source_files``: ``(y, theta)`` filenames

        Both grid axes are sorted in increasing order, independent of the
        original snake sweep order.
    """
    if pos_scale == 0 or angle_scale == 0:
        raise ValueError("pos_scale and angle_scale must be nonzero.")

    directory = Path(directory)
    pattern = "init_*_finalTip_x=*_y=*.csv"
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {directory}")

    filename_re = re.compile(
        r"^init_([01]+)_finalTip_x=.*_y=.*\.csv$"
    )
    records_by_init: dict[str, list[dict[str, Any]]] = {}
    required_columns = {
        "upd_x_tip", "upd_y_tip", "upd_tip_angle", "buckle_arr_update",
    }

    for path in files:
        match = filename_re.fullmatch(path.name)
        if match is None:
            continue

        with open(path, newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            missing_columns = required_columns - set(reader.fieldnames or [])
            if missing_columns:
                raise ValueError(
                    f"{path.name} is missing required columns: {sorted(missing_columns)}"
                )
            final_rows = deque(reader, maxlen=1)
        if not final_rows:
            raise ValueError(f"{path.name} contains no transition rows.")

        final_row = final_rows[0]
        init_bits = match.group(1)
        records_by_init.setdefault(init_bits, []).append({
            "x": round(float(final_row["upd_x_tip"]) / pos_scale, 8),
            "y": round(float(final_row["upd_y_tip"]) / pos_scale, 8),
            "theta": round(float(final_row["upd_tip_angle"]) / angle_scale, 8),
            "buckle": helpers_builders.buckle_cell_to_array(
                final_row["buckle_arr_update"]
            ).astype(np.int32),
            "file": path.name,
        })

    maps: dict[str, dict[str, Any]] = {}
    for init_bits, records in sorted(records_by_init.items()):
        y_values = np.asarray(sorted({record["y"] for record in records}), dtype=float)
        theta_values = np.asarray(
            sorted({record["theta"] for record in records}), dtype=float
        )
        expected_files = len(y_values) * len(theta_values)
        if len(records) != expected_files:
            raise ValueError(
                f"Initial buckle {init_bits} has {len(records)} files but its "
                f"{len(y_values)} x {len(theta_values)} grid requires {expected_files}."
            )

        buckle_shape = records[0]["buckle"].shape
        buckle_matrix = np.empty(
            (len(y_values), len(theta_values), *buckle_shape), dtype=np.int32
        )
        grid_points = np.empty((len(y_values), len(theta_values), 3), dtype=float)
        source_files = np.empty((len(y_values), len(theta_values)), dtype=object)
        occupied = np.zeros((len(y_values), len(theta_values)), dtype=bool)
        y_index = {value: i for i, value in enumerate(y_values)}
        theta_index = {value: i for i, value in enumerate(theta_values)}

        for record in records:
            if record["buckle"].shape != buckle_shape:
                raise ValueError(
                    f"Inconsistent buckle shape in {record['file']}: "
                    f"expected {buckle_shape}, got {record['buckle'].shape}."
                )
            yi = y_index[record["y"]]
            ti = theta_index[record["theta"]]
            if occupied[yi, ti]:
                raise ValueError(
                    f"Duplicate grid point y={record['y']}, theta={record['theta']} "
                    f"for initial buckle {init_bits}."
                )
            occupied[yi, ti] = True
            buckle_matrix[yi, ti] = record["buckle"]
            grid_points[yi, ti] = [record["x"], record["y"], record["theta"]]
            source_files[yi, ti] = record["file"]

        if not np.all(occupied):
            raise ValueError(f"Initial buckle {init_bits} has missing grid points.")

        buckle_ids = np.asarray([
            helpers_builders.buckle_to_index(buckle.reshape(-1))
            for buckle in buckle_matrix.reshape(-1, *buckle_shape)
        ], dtype=np.int32).reshape(len(y_values), len(theta_values))

        maps[init_bits] = {
            "buckle_matrix": buckle_matrix,
            "buckle_grid_frames": buckle_matrix.reshape(
                len(y_values) * len(theta_values), *buckle_shape
            ),
            "buckle_ids": buckle_ids,
            "grid_points": grid_points,
            "x_values": grid_points[:, 0, 0],
            "y_values": y_values,
            "theta_values": theta_values,
            "y_num": len(y_values),
            "theta_num": len(theta_values),
            "source_files": source_files,
            "y_scale": float(pos_scale),
        }

    return maps


def export_tip_grid_transition_csvs(output_dir: str | Path, Sprvsr: "SupervisorClass",
                                    State_grid: "StateClass", init_buckle: NDArray,
                                    *, include_flat_target: bool = False) -> list[Path]:
    """Export one ordered trajectory CSV for each sweep target.

    Polar-grid files contain every accepted command for one spatial grid point:
    the completed move to that point, every configured angle, and any fractional
    refinement steps. Diagonal-grid files retain the true initial state, only
    distinct intermediate buckle states, and the final target.

    Parameters
    ----------
    output_dir : str or Path
        Directory in which the per-target CSV files are created.
    Sprvsr : SupervisorClass
        Supervisor populated by ``tip_diag_buckle_sweep`` or
        ``tip_grid_buckle_sweep``.
    State_grid : StateClass
        State history returned by the sweep.
    init_buckle : ndarray
        Buckle state from which every independent grid trial starts.
    include_flat_target : bool, default=False
        Also export the leading flat-pose target used to initialize the sweep.

    Returns
    -------
    list[Path]
        Paths of the files written, in sweep order.
    """
    def arr_to_json(arr: np.ndarray) -> str:
        return json.dumps(np.asarray(arr).tolist(), separators=(",", ":"))

    def filename_number(value: float) -> str:
        value = 0.0 if np.isclose(value, 0.0, atol=5e-10) else float(value)
        return np.format_float_positional(value, precision=8, unique=True, trim="-")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    is_polar_grid = Sprvsr.dataset_sampling == "tip_grid_sweep"

    final_indices = np.asarray(Sprvsr.tip_grid_target_indices, dtype=int)
    start_indices = np.asarray(Sprvsr.tip_grid_target_start_indices, dtype=int)
    fractions = np.asarray(Sprvsr.tip_grid_path_fraction_in_t, dtype=float)
    target_pos = np.asarray(Sprvsr.tip_grid_base_pos_in_t, dtype=float)
    target_angle = np.asarray(Sprvsr.tip_grid_base_angle_in_t, dtype=float)
    buckle_history = np.moveaxis(np.asarray(State_grid.buckle_in_t), 2, 0)
    pos_history = np.moveaxis(np.asarray(State_grid.pos_arr_in_t), 2, 0)

    if not (len(start_indices) == len(final_indices) == len(target_pos) == len(target_angle)):
        raise ValueError("Tip-grid target metadata have inconsistent lengths.")

    init_buckle_arr = np.asarray(init_buckle, dtype=int).reshape(buckle_history.shape[1:])
    init_buckle_str = correct_buckle_string(init_buckle_arr)
    initial_pos = (
        np.asarray(Sprvsr.tip_grid_initial_pos, dtype=float)
        if is_polar_grid else pos_history[final_indices[0]]
    )
    initial_tip_pos = initial_pos[-1] if is_polar_grid else target_pos[0]
    initial_tip_angle = 0.0 if is_polar_grid else target_angle[0]
    first_target = 0 if is_polar_grid or include_flat_target else 1
    written: list[Path] = []

    for target_i in range(first_target, len(final_indices)):
        start = int(start_indices[target_i])
        final = int(final_indices[target_i])
        if is_polar_grid:
            row_indices: list[int | None] = list(range(start, final + 1))
        else:
            final_buckle = buckle_history[final]

            # Retain the ordered buckle changes needed to describe the path.
            change_indices: list[int] = []
            previous = init_buckle_arr
            for frame_i in range(start, final + 1):
                buckle = buckle_history[frame_i]
                if np.array_equal(buckle, previous):
                    continue
                change_indices.append(frame_i)
                previous = buckle
            nonfinal_changes = [
                frame_i for frame_i in change_indices
                if not np.array_equal(buckle_history[frame_i], final_buckle)
            ]
            last_middle = nonfinal_changes[-1] if nonfinal_changes else -1
            middle_indices = [frame_i for frame_i in change_indices if frame_i <= last_middle]
            row_indices = [None, *middle_indices, final]

        x_name = filename_number(target_pos[target_i, 0] * Sprvsr.convert_pos)
        y_name = filename_number(target_pos[target_i, 1] * Sprvsr.convert_pos)
        if is_polar_grid:
            path = output_dir / (
                f"init_{init_buckle_str}_finalTip_x={x_name}_y={y_name}.csv"
            )
        else:
            theta_name = filename_number(target_angle[target_i] * Sprvsr.convert_angle)
            path = output_dir / (
                f"init_{init_buckle_str}_finalTip_x={x_name}_y={y_name}_theta={theta_name}.csv"
            )

        rows = []
        for event_t, frame_i in enumerate(row_indices):
            if frame_i is None:
                rows.append({
                    "t": event_t,
                    "path_fraction": 0.0,
                    "is_final": False,
                    "upd_x_tip": float(initial_tip_pos[0] * Sprvsr.convert_pos),
                    "upd_y_tip": float(initial_tip_pos[1] * Sprvsr.convert_pos),
                    "upd_tip_angle": float(initial_tip_angle * Sprvsr.convert_angle),
                    "final_pos_update": arr_to_json(initial_pos * Sprvsr.convert_pos),
                    "Fx_update": 0.0,
                    "Fy_update": 0.0,
                    "buckle_arr_update": arr_to_json(init_buckle_arr),
                })
                continue

            rows.append({
                "t": event_t,
                "path_fraction": float(fractions[frame_i]),
                "is_final": frame_i == final,
                "upd_x_tip": float(Sprvsr.tip_pos_update_in_t[frame_i, 0] * Sprvsr.convert_pos),
                "upd_y_tip": float(Sprvsr.tip_pos_update_in_t[frame_i, 1] * Sprvsr.convert_pos),
                "upd_tip_angle": float(Sprvsr.tip_angle_update_in_t[frame_i] * Sprvsr.convert_angle),
                "final_pos_update": arr_to_json(pos_history[frame_i] * Sprvsr.convert_pos),
                "Fx_update": float(State_grid.Fx_in_t[frame_i] * Sprvsr.convert_F),
                "Fy_update": float(State_grid.Fy_in_t[frame_i] * Sprvsr.convert_F),
                "buckle_arr_update": arr_to_json(buckle_history[frame_i]),
            })

        pd.DataFrame(rows).to_csv(path, index=False)
        written.append(path)

    Sprvsr.tip_grid_transition_csv_mode = (
        "all_accepted_steps_xy_filename" if is_polar_grid else "buckle_changes_only"
    )
    return written


# ---------------------------------------------------------------
# Post-processing files
# ---------------------------------------------------------------
def loss_from_filename(file: Path):
    return float(re.search(r"final_loss_(.*?)_init_", file.stem).group(1))


def successful_train_times(folder: str | Path, thresh: float = 1e-6,
                           omit_inverted: bool = False, omit_intersect: bool = False) -> pd.DataFrame:
    """
    Summarize final training times for successful training CSVs in a folder.

    Parameters
    ----------
    folder : str | Path
        Folder containing ``final_loss_*.csv`` training exports.
    thresh : float, default=1e-6
        Maximum final loss for a run to count as successful.
    omit_inverted : bool, default=False
        Omit files containing ``"_inverted"`` in the filename.
    omit_intersect : bool, default=False
        Omit files containing ``"_intersect"`` in the filename.

    Returns
    -------
    pandas.DataFrame
        One row per successful CSV, with final loss, final ``t`` value, row count,
        and filename flags.

    Notes
    -----
    This reports the algorithm training-step time stored in the CSV ``t`` column,
    not wall-clock runtime. Per-run wall-clock time is written to the log files.
    """
    folder = Path(folder)
    files = sorted(folder.glob("final_loss_*.csv"))
    if omit_inverted:
        files = [file for file in files if "_inverted" not in file.stem]
    if omit_intersect:
        files = [file for file in files if "_intersect" not in file.stem]
    if not files:
        raise FileNotFoundError(f"No files matching 'final_loss_*.csv' in {folder}")

    rows: list[dict[str, Any]] = []

    for file in files:
        final_loss = loss_from_filename(file)
        if final_loss >= thresh:
            continue

        df = pd.read_csv(file, usecols=["t"])
        if df.empty:
            continue

        rows.append({
            "file": file.name,
            "final_loss": final_loss,
            "train_time": float(df["t"].iloc[-1]),
            "row_count": int(len(df)),
            "inverted": "_inverted" in file.stem,
            "intersect": "_intersect" in file.stem,
        })

    if not rows:
        raise ValueError(f"No successful training CSVs found in {folder} with final loss < {thresh}")

    return pd.DataFrame(rows).sort_values("train_time", ascending=False, ignore_index=True)


def average_successful_train_time(folder: str | Path, thresh: float = 1e-6,
                                  omit_inverted: bool = False, omit_intersect: bool = False) -> float:
    """
    Average the final training time over successful training CSVs in a folder.

    Parameters
    ----------
    folder : str | Path
        Folder containing ``final_loss_*.csv`` training exports.
    thresh : float, default=1e-6
        Maximum final loss for a run to count as successful.
    omit_inverted : bool, default=False
        Omit files containing ``"_inverted"`` in the filename.
    omit_intersect : bool, default=False
        Omit files containing ``"_intersect"`` in the filename.

    Returns
    -------
    float
        Mean final ``t`` value across successful runs.
    """
    df = successful_train_times(folder=folder, thresh=thresh, omit_inverted=omit_inverted,
                                omit_intersect=omit_intersect)
    train_times = df["train_time"].to_numpy(dtype=float)

    return float(np.mean(train_times))


def build_loss_columns(folder: str | Path, old: bool = False, omit_inverted: bool = False,
                       log_norm: bool = True, save_path: Optional[str | Path] = None,
                       include_symm: bool = False) -> Tuple[NDArray[np.float64],
                                                           NDArray[np.float64], NDArray]:
    """
    Read training CSVs in a folder and plot initial/final MSE loss as two color columns.

    Parameters
    ----------
    folder        : str | Path
        Folder containing ``final_loss_*.csv`` training exports.
    old           : bool, default=False
        Use old filename convention where desired buckle appears as ``desiredXXXX``.
    omit_inverted : bool, default=False
        Omit files containing ``"_inverted"`` in the filename.
    log_norm      : bool, default=True
        If True, use logarithmic color scaling for positive losses.
    save_path     : str | Path | None
        Optional path for saving the figure.
    include_symm  : bool, default=False
        If True, assign each task the elementwise minimum initial/final loss
        and Hamming distance of that task and its reciprocal task. A reciprocal
        task is obtained by flipping every bit in both the initial and desired
        buckle states.

    Returns
    -------
    loss_columns : ndarray, shape (N, 2)
        ``loss_columns[:, 0]`` is initial ``loss_MSE`` and
        ``loss_columns[:, 1]`` is final ``loss_MSE``.
    Hamming_columns : ndarray, shape (N, 2)
        Initial and final Hamming distance. Missing CSV columns are returned as ``nan``.
    buckle_pairs : ndarray, shape (N, 2)
        String buckle labels ``[initial_buckle, desired_buckle]`` for every row.
    """
    folder = Path(folder)
    files = sorted(folder.glob("final_loss_*.csv"))
    if omit_inverted:
        files = [file for file in files if "_inverted" not in file.stem]
    if not files:
        raise FileNotFoundError(f"No files matching 'final_loss_*.csv' in {folder}")

    records: list[tuple[int, int, list[float], list[float], list[str]]] = []

    for file in files:
        name = file.stem
        init_match = re.search(r"init_([01]+)", name)
        desired_match = re.search(r"desired([01]+)", name) if old else re.search(r"desired_([01]+)", name)
        if init_match is None or desired_match is None:
            continue

        try:
            df = pd.read_csv(file)
        except pd.errors.EmptyDataError:
            init_bits = init_match.group(1)
            desired_bits = desired_match.group(1)
            records.append((int(init_bits, 2), int(desired_bits, 2),
                            [1.0, loss_from_filename(file)], [np.nan, np.nan],
                            [init_bits, desired_bits]))
            continue
        if df.empty:
            continue

        if "loss_MSE" in df.columns:
            loss_MSE = df["loss_MSE"].to_numpy(dtype=float)
        else:
            loss_cols = [c for c in df.columns if re.fullmatch(r"loss_\d+", c)]
            loss_cols = sorted(loss_cols, key=lambda c: int(c.split("_")[1]))
            if len(loss_cols) == 0:
                continue
            loss_MSE = np.mean(df[loss_cols].to_numpy(dtype=float)**2, axis=1)

        if "Hamming_distance" in df.columns:
            Hamming = df["Hamming_distance"].to_numpy(dtype=float)
        else:
            Hamming = np.full(loss_MSE.shape, np.nan, dtype=float)

        init_bits = init_match.group(1)
        desired_bits = desired_match.group(1)
        init_idx = 1 if loss_MSE.size > 1 else 0
        records.append((int(init_bits, 2), int(desired_bits, 2),
                        [float(loss_MSE[init_idx]), float(loss_MSE[-1])],
                        [float(Hamming[init_idx]), float(Hamming[-1])],
                        [init_bits, desired_bits]))

    if not records:
        raise ValueError(f"No readable training loss files found in {folder}")

    records = sorted(records, key=lambda record: (record[0], record[1]))
    losses = [record[2] for record in records]
    Hammings = [record[3] for record in records]
    buckle_pairs = [record[4] for record in records]

    loss_columns = np.asarray(losses, dtype=float)
    Hamming_columns = np.asarray(Hammings, dtype=float)
    buckle_pairs_arr = np.asarray(buckle_pairs, dtype=str)

    if include_symm:
        task_losses: dict[tuple[str, str], NDArray[np.float64]] = {}
        task_Hammings: dict[tuple[str, str], NDArray[np.float64]] = {}
        for pair, loss, Hamming in zip(buckle_pairs, loss_columns, Hamming_columns):
            task = (pair[0], pair[1])
            if task in task_losses:
                task_losses[task] = np.fmin(task_losses[task], loss)
                task_Hammings[task] = np.fmin(task_Hammings[task], Hamming)
            else:
                task_losses[task] = loss.copy()
                task_Hammings[task] = Hamming.copy()

        for index, pair in enumerate(buckle_pairs):
            n_bits = len(pair[0])
            reciprocal_indices = helpers_builders.reciprocal_transition(
                (int(pair[0], 2), int(pair[1], 2)), n_bits)
            reciprocal_task = tuple(format(state, f"0{n_bits}b")
                                    for state in reciprocal_indices)
            reciprocal_loss = task_losses.get(reciprocal_task)
            if reciprocal_loss is not None:
                loss_columns[index] = np.fmin(loss_columns[index], reciprocal_loss)
                Hamming_columns[index] = np.fmin(Hamming_columns[index],
                                                 task_Hammings[reciprocal_task])

    return loss_columns, Hamming_columns, buckle_pairs_arr


def build_success_matrix(folder: Path, old: bool = False, N: int = 16, near_miss: bool = False,
                         symmetry: bool = False, omit_inverted: bool = False,
                         find_symmetrical: bool = False) -> Tuple[NDArray, NDArray]:
    """
    Build matrix marking which runs successded (M) alongside which runs had self-intersections (M_flag).

    Parameters:
    -----------
    folder        : path to folder where all the export_training.csv files are at, starting with "loss=..."
    old           : boolean whether to use old files or not, new are since Mar2026.
    N             : int, total number of states, 2^hinges
    omit_inverted : omit files ending with "_inverted.csv"

    Returns:
    --------
    M      : (N, N) success matrix
        0 - successful training
        1 - didn't train on this path
        2 - unsuccessful training
    M_flag : (N, N) flag matrix
        1 where the run file exists and is flagged as intersecting,
        0 otherwise.
    """
    M = np.zeros((N, N)) + 1.0
    M_flag = np.zeros((N, N), dtype=int)
    M_flip = np.zeros((N, N), dtype=int)
    B = np.zeros((N, N)) + 1.0

    thresh = 10e-2 if near_miss else 1e-6

    for file in folder.glob("final_loss_*.csv"):
        name = file.stem

        # omit inverted
        if omit_inverted and "_inverted" in name:
            continue

        loss = loss_from_filename(file)

        # buckles
        init_bits = re.search(r"init_([01]+)", name).group(1)
        desired_bits = (re.search(r"desired([01]+)", name).group(1) if old
                        else re.search(r"desired_([01]+)", name).group(1))
        init = [int(ch) for ch in init_bits]
        desired = [int(ch) for ch in desired_bits]

        # indices in matrix from buckles
        i = helpers_builders.buckle_to_index(init)
        j = helpers_builders.buckle_to_index(desired)

        # run already successfull
        if B[i, j] == 0:
            continue

        # boolean whether file is successfull run or not
        success: bool = loss < thresh

        # flag of intersection
        flagged_int = "_intersect" in name
        if find_symmetrical:
            flagged_flip = is_symmetrical_reached(file=file)
            if flagged_flip:
                success = True
        else:
            flagged_flip = "_flip_chain" in name

        if success:
            B[i, j] = 0
            M[i, j] = 0
            M_flag[i, j] = int(flagged_int)   # flag for intersection in successful run
            M_flip[i, j] = int(flagged_flip)  # flag for chain symmetrical to desired
        else:
            if M[i, j] == 2 and flagged_int:
                M_flag[i, j] = int(flagged_int)   # flag for intersection in both unsuccessful runs
                M_flip[i, j] = int(flagged_flip)  # flag for chain symmetrical to desired
            M[i, j] = 2

        # symmetry
        if symmetry:
            B[N-1-i, N-1-j] = B[i, j]
            M[N-1-i, N-1-j] = M[i, j]

    return M, M_flag, M_flip


def shortest_success_paths(M: np.ndarray):
    """
    Treat direct successes (M==0) as directed edges.
    Returns
    -------
    reachable : (N,N) bool
        reachable[i,j] is True iff j can be reached from i through direct-success edges.
    next_hop : (N,N) int
        for path reconstruction; -1 means unreachable.
    dist : (N,N) int
        number of edges in shortest path; large value if unreachable.
    """
    N = M.shape[0]
    reachable = np.zeros((N, N), dtype=bool)
    next_hop = -np.ones((N, N), dtype=int)
    dist = np.full((N, N), np.inf)

    # self reachability
    for i in range(N):
        reachable[i, i] = True
        next_hop[i, i] = i
        dist[i, i] = 0

    # direct edges = direct successful runs
    for i in range(N):
        for j in range(N):
            if M[i, j] == 0:
                reachable[i, j] = True
                next_hop[i, j] = j
                dist[i, j] = 1

    # Floyd-Warshall for transitive closure + shortest path
    for k in range(N):
        for i in range(N):
            if not reachable[i, k]:
                continue
            for j in range(N):
                if not reachable[k, j]:
                    continue
                cand = dist[i, k] + dist[k, j]
                if cand < dist[i, j]:
                    reachable[i, j] = True
                    dist[i, j] = cand
                    next_hop[i, j] = next_hop[i, k]

    return reachable, next_hop, dist


def reconstruct_path(i: int, j: int, next_hop: np.ndarray):
    """
    Return path [i, ..., j] as indices.
    Empty list if unreachable.
    """
    if next_hop[i, j] == -1:
        return []

    path = [i]
    cur = i
    while cur != j:
        cur = next_hop[cur, j]
        if cur == -1:
            return []
        path.append(cur)

        # safety against unexpected loops
        if len(path) > next_hop.shape[0] + 1:
            raise RuntimeError("Path reconstruction got stuck in a loop.")

    return path


def corrected_success_matrix(M: np.ndarray):
    """
    Add a new code:
    3 - indirect success via one or more successful intermediate states
    """
    reachable, next_hop, dist = shortest_success_paths(M)
    M_corr = M.copy()

    N = M.shape[0]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            # keep direct success as 0
            if M[i, j] == 0:
                continue

            # if not direct success, but reachable through a path of length >=2
            if reachable[i, j] and dist[i, j] >= 2:
                M_corr[i, j] = 3

    return M_corr, next_hop, dist


def print_indirect_paths(M_corr: np.ndarray, next_hop: np.ndarray, only_fixed=None):
    """
    Print all newly rescued transitions.
    only_fixed: optional iterable of (i,j) pairs to print only selected cases.
    """
    N = M_corr.shape[0]

    pairs = []
    for i in range(N):
        for j in range(N):
            if M_corr[i, j] == 3:
                pairs.append((i, j))

    if only_fixed is not None:
        pairs = [p for p in pairs if p in set(only_fixed)]

    for i, j in pairs:
        path = reconstruct_path(i, j, next_hop)
        path_str = " -> ".join(helpers_builders.index_to_buckle(k) for k in path)
        print(f"{helpers_builders.index_to_buckle(i)} -> {helpers_builders.index_to_buckle(j)}  via  {path_str}")


def get_pathway_between_states(init_state: str, desired_state: str, next_hop: np.ndarray):
    """
    Example:
        get_pathway_between_states("0011", "1110", next_hop)
    """
    i = int(init_state, 2)
    j = int(desired_state, 2)
    path = reconstruct_path(i, j, next_hop)
    if not path:
        return None
    return [helpers_builders.index_to_buckle(k) for k in path]


def is_symmetrical_reached(file: Path) -> bool:

    name = file.name

    target_str = re.search(r"desired_([01]+)", name).group(1)
    target_arr = helpers_builders.buckle_cell_to_array(target_str)
    symmetrical_target = -target_arr
    # print('target_arr=', target_arr)

    df = pd.read_csv(file)

    buckles = df["buckle_arr_update"].apply(lambda s: np.array(json.loads(s), dtype=int))

    reached = buckles.apply(lambda a: np.array_equal(a, symmetrical_target)).any()

    if reached:
        return True

    return False


# ---------------------------------------------------------------
# Transition diagram
# ---------------------------------------------------------------
def _infer_buckle_n_bits(folder: Path, omit_inverted: bool = False) -> int:
    """
    Infer the number of buckle bits from the first transition CSV in a folder.
    """
    file_patterns = ("final_loss_*.csv", "init_*_finalTip_x=*_y=*.csv",
                     "tip_diag_buckle_sweep_*.csv", "tip_grid_buckle_sweep_*.csv", "tip_buckle*.csv")
    files = sorted({file for pattern in file_patterns for file in folder.glob(pattern)})
    if omit_inverted:
        files = [f for f in files if "_inverted" not in f.stem]
    if not files:
        raise FileNotFoundError(f"No files matching {file_patterns} in {folder}")

    df = pd.read_csv(files[0], nrows=1)
    if "buckle_arr_update" in df:
        return int(helpers_builders.buckle_cell_to_array(df["buckle_arr_update"].iloc[0], keep_2d=False).size)
    return len(helpers_builders.infer_buckle_columns(df))


def _as_bool(value: bool | str) -> bool:
    """Accept booleans and notebook-friendly string booleans."""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def buckle_transitions(folder: str | Path, only_init_and_final_buckles: bool = False, omit_inverted: bool = False,
                       transition_mode: str = "hamming", reciprocity: bool | str = False):
    """
    Go over all final_loss_*.csv files and extract directed buckle transitions.

    Parameters
    ----------
    folder                      : path, all csv run files, from every init to every desired
    only_init_and_final_buckles : bool, True = transition is only from initial to final (not necessarily the desired)
                                  desired transition colored Cyan, undesired colored purple
    omit_inverted               : bool, True = omit files containing ``_inverted`` in their stem
    transition_mode             : str, "hamming" checks missing one-bit transitions; "ring" checks all-to-all transitions
    reciprocity                 : bool, True = add each transition count to its bitwise sign-opposite transition

    Returns
    -------
    transitions          : Counter[(src, dst)] = number of times observed across all files
    per_file_transitions : dict[file_name, list[(src, dst)]]
    per_file_loss        : dict[file_name, float]
    edge_zero_loss_count : Counter[(src, dst)] = number of zero-loss files on this edge
    missing_edges        : ???
    """
    folder = Path(folder)
    if transition_mode not in {"hamming", "ring", "all_to_all"}:
        raise ValueError("transition_mode must be 'hamming', 'ring', or 'all_to_all'")

    transitions, per_file_transitions, per_file_loss, edge_zero_loss_count = helpers_builders.build_transition_counts(folder,
                                                                                                                      only_init_and_final_buckles=only_init_and_final_buckles,
                                                                                                                      omit_inverted=omit_inverted)

    n_bits = _infer_buckle_n_bits(folder, omit_inverted=omit_inverted)
    reciprocity = _as_bool(reciprocity)
    if reciprocity:
        transitions = helpers_builders.add_reciprocal_transition_counts(transitions, n_bits)
        edge_zero_loss_count = helpers_builders.add_reciprocal_transition_counts(edge_zero_loss_count, n_bits)

    observed_edges = set(transitions.keys())
    if transition_mode in {"ring", "all_to_all"}:
        missing_edges = [edge for edge in helpers_builders.all_possible_transitions(n_bits) if edge not in observed_edges]
    else:
        missing_edges = [edge for edge in helpers_builders.all_possible_transitions(n_bits) if
                         edge not in observed_edges and helpers_builders.hamming_distance_int(*edge) == 1]

    print(f"Found {len(per_file_transitions)} files")
    print(f"Found {sum(transitions.values())} total transitions")
    print(f"Found {len(transitions)} unique directed transitions\n")
    if reciprocity:
        print("Reciprocity enabled: transition counts include bitwise sign-opposite partners\n")

    print("Top transitions:")
    for (a, b), c in transitions.most_common(20):
        print(f"{helpers_builders.index_to_buckle(a, n_bits)} -> {helpers_builders.index_to_buckle(b, n_bits)}: {c}")

    return transitions, per_file_transitions, per_file_loss, edge_zero_loss_count, missing_edges


# ---------------------------------------------------------------
# Build functions from file
# ---------------------------------------------------------------
def build_torque_and_k_from_file(path: str, *, contact: bool = True, angles_in_degrees: bool = True,
                                 savgol_window: Optional[int] = None, 
                                 contact_scale: float = 1e2,) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray,
                                                                       Callable[[jnp.ndarray], jnp.ndarray],
                                                                       Callable[[jnp.ndarray], jnp.ndarray]]:
    """
    Load torque–angle measurement from file and construct JAX-compatible interpolation funcs for torque & stiffness
    Stiffness ``k = dτ/dθ``

    Parameters
    ----------
    path              - str, path to the text/CSV file containing two columns: angle and torque.
    contact           - bool, If True, extend torque function outside measured range for contact-induced divergence.
    angles_in_degrees - bool, If True, convert angles from degrees to radians.
    savgol_window     - int, window length for Savitzky–Golay smoothing of stiffness curve. Must be odd int>2
    contact_scale     - float, contact scaling factor relative to maximal measured torque.

    Returns
    -------
    theta_grid      - jnp.ndarray, shape (N,), sorted vector of angles (radians).
    torque_grid     - jnp.ndarray, shape (N,), torque samples over theta.
    k_grid          - jnp.ndarray, shape (N,), local stiffness as numeric derivative of torque w.r.t. theta.
    torque_of_theta - callable, JAX function theta -> torque interpolation including diverging forces at contact.
    k_of_theta      - callable, JAX function theta -> stiffness interpolation.

    Notes
    -----
    - Negative stiffness values clipped to small positive value (``1e-3``).
    - Contact occurs outside the measured range.
    """
    # ------ load as numpy, sort, unique------
    try:
        data = np.loadtxt(path)                # shape (N, 2)
    except ValueError:
        data = np.loadtxt(path, delimiter=',')
    theta = data[:, 0]
    tau = data[:, 1]

    if path in {"single_hinge_files/Roie_metal_singleMylar_short.csv",
                "single_hinge_files/Stress_Strain_steel_1myl1tp_short.csv",
                "single_hinge_files/Stress_Strain_1myl1tp_otherEnd_short.csv"}:  # flip axes
        tau = -tau

    # degrees -> radians if needed
    if angles_in_degrees:
        theta = np.deg2rad(theta)

    # sort & unique (interp requires monotonic x)
    order = np.argsort(theta)
    theta = theta[order]
    tau = tau[order]
    # collapse duplicates (if any)
    theta_u, idx = np.unique(theta, return_index=True)
    tau_u = tau[idx]

    # ------ numeric derivative: k = d(tau)/d(theta) ------
    k = np.gradient(tau_u, theta_u)

    # optional light smoothing of k (pure NumPy, outside JAX)
    if savgol_window is not None and savgol_window > 2 and savgol_window % 2 == 1:
        try:
            k = savgol_filter(k, window_length=savgol_window, polyorder=4, mode="interp")
        except Exception:
            print('SciPy isnt available, just skip smoothing')

    # ------ JAX arrays ------
    theta_grid = jnp.asarray(theta_u, dtype=jnp.float32)
    torque_grid = jnp.asarray(tau_u, dtype=jnp.float32)
    k_grid = jnp.asarray(k, dtype=jnp.float32)
    k_grid = k_grid.at[k_grid < 0].set(10e-4)  # for numerical stability, singular point of experimental negative k

    # ----- linear interpolators (JAX) ------
    def torque_of_theta(theta_query: jnp.ndarray) -> jnp.ndarray:
        # masks for outside vs inside range
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

    def _clamp(x, xmin, xmax):
        return jnp.clip(x, xmin, xmax)

    return theta_grid, torque_grid, k_grid, torque_of_theta, k_of_theta


# -----------------------------
# parallel jobs
# -----------------------------
def already_done_in_dir(job, prev_dir: Path, H: int):
    """
    Roie - document

    parameters:
    H      - number of hinges
    """
    init = correct_buckle_string(np.array(job["init_buckle_tup"]).reshape(H, 1))
    des = correct_buckle_string(np.array(job["desired_buckle_tup"]).reshape(H, 1))

    pattern = f"*init_{init}_desired_{des}*"
    # return any(p.is_file() and not (p.name.startswith("log_") or p.name.startswith("gif_")) for p in Path(prev_dir).glob(pattern))
    return any(p.is_file() and p.suffix == ".csv" for p in Path(prev_dir).glob(pattern))


# -----------------------------
# File helpers
# -----------------------------
def correct_buckle_string(buckle_arr: NDArray):
    buckle = helpers_builders.jax2numpy(buckle_arr, dtype=int)
    buckle = copy.copy(buckle)
    buckle[buckle == -1] = 0
    buckle_str = ''.join(buckle.reshape(-1).astype(str))
    return buckle_str


# # ==========
# # NOT IN USE
# # ==========
# def export_stress_strain_sim(Sprvsr: "SupervisorClass", Fx_afo_pos: NDArray[np.float64], Fy_afo_pos: NDArray[np.float64], 
#                              L: float, buckle_arr: NDArray[np.int], filename: str = None) -> None:
#     tip_pos_in_t = Sprvsr.tip_pos_in_t * Sprvsr.convert_pos
#     tip_angle_in_t = Sprvsr.tip_angle_in_t * Sprvsr.convert_angle
#     Fx_afo_pos = Fx_afo_pos * Sprvsr.convert_F
#     Fy_afo_pos = Fy_afo_pos * Sprvsr.convert_F

#     # --- build pandas dataframe ---
#     df = pd.DataFrame({
#         "x_tip": tip_pos_in_t[:, 0],
#         "y_tip": tip_pos_in_t[:, 1],
#         "tip_angle_deg": tip_angle_in_t,
#         "F_x": Fx_afo_pos,
#         "F_y": Fy_afo_pos,
#     })
#     if filename is not None:
#         pass 
#     else:
#         filename = f"L={L}_buckle{buckle_arr.reshape(-1)}.csv"  # filename example "L=1_buckle1111.csv"
#     out_path = Path(filename)
#     df.to_csv(out_path, index=False)

# def import_stress_strain_sim_and_plot(path: str, plot: bool = False) -> df:
#     sim_df = pd.read_csv(path)   # assumes the header row is in the file
#     if plot:
#         plt.plot(sim_df['x_tip'], sim_df['Fx'])
#         plt.xlabel('tip pos')
#         plt.ylabel('Fx')
#         plt.show()
#     return sim_df


# def import_stress_strain_exp_and_plot(path: str, plot: bool = True) -> None:
#     exp_df = pd.read_csv(path)   # assumes the header row is in the file
#     if plot:
#         plt.plot(exp_df['Position (mm)'], exp_df['Load2 (N)'])
#         plt.xlabel('tip pos')
#         plt.ylabel('Fx')
#         plt.show()
#     return exp_df
