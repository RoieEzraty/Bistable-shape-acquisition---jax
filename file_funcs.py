from __future__ import annotations

import csv
import copy
import re
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
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
    stretch_factor : Optional[float], Optional scaling applied to x and y positions,
                                      for rescaling experimental trajectories.

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


def load_training(path: str, stretch_factor: Optional[float] = None) -> Tuple[NDArray[np.float64], NDArray[np.float64],
                                                                              NDArray[np.float64], NDArray[np.float64],
                                                                              NDArray[np.float64], NDArray[np.float64]]:
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
    stretch_factor : Optional[float], Optional scaling applied to x and y positions,
                                      for rescaling experimental trajectories.

    Returns
    -------

    L        - ndarray, shape (T, loss_dim), loss_0/loss_1/(optional loss_2...)
    B        - ndarray, shape (T, H, S), measured buckle arrays
    P_update - ndarray, shape (T, 2), updated tip positions
    A_update - ndarray, shape (T,), updated tip angles in radians

    Notes
    -----
    - The loader accepts multiple possible column names for compatibility
      with different datasets (e.g. `"x_tip"`, `"pos_x"`, `"Px"`).
    - Angles always returned in **radians** when `mod="arrays"`.
    """
    L, B = [], []
    P_meas, P_update = [], []
    tip_update, angle_update = [], []

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
            Buckle, _ = helpers_builders._get_first_in_file(r, ["buckle_arr_meas"], name="buckle_arr_meas",
                                                            type="NDArray")
            B.append(Buckle)

            # ---- tip position / angle ----
            X_update, _ = helpers_builders._get_first_in_file(r, ["upd_x_tip"], name="upd_x_tip")
            Y_update, _ = helpers_builders._get_first_in_file(r, ["upd_y_tip"], name="upd_y_tip")
            Angle_update, theta_key = helpers_builders._get_first_in_file(r, ["upd_tip_angle"],
                                                                          name="upd_tip_angle")

            if stretch_factor is not None:
                X_update *= stretch_factor
                Y_update *= stretch_factor

            # export_training_csv writes upd_tip_angle in degrees, so convert to radians
            if theta_key != "tip_angle_rad":
                Angle_update = np.deg2rad(Angle_update)

            tip_update.append([X_update, Y_update])
            angle_update.append(Angle_update)

            # ------ Full positions ------
            pos_meas, _ = helpers_builders._get_first_in_file(r, ["final_pos_meas"], name="final_pos_meas",
                                                              type="NDArray", allow_missing=True)
            pos_update, _ = helpers_builders._get_first_in_file(r, ["final_pos_update"], name="final_pos_update",
                                                                type="NDArray", allow_missing=True)

            if pos_meas is not None:
                pos_meas = np.asarray(pos_meas, dtype=float)
                if stretch_factor is not None:
                    pos_meas *= stretch_factor
                P_meas.append(pos_meas)

            if pos_update is not None:
                pos_update = np.asarray(pos_update, dtype=float)
                if stretch_factor is not None:
                    pos_update *= stretch_factor
                P_update.append(pos_update)

    L = np.asarray(L, dtype=float)

    B = np.stack(B, axis=0)          # (T, H, 1)
    B = np.moveaxis(B, 0, -1)        # (H, 1, T)

    P_meas = np.asarray(P_meas, dtype=float)
    P_update = np.asarray(P_update, dtype=float)

    tip_update = np.asarray(tip_update, dtype=float)
    angle_update = np.asarray(angle_update, dtype=float)

    return L, B, P_meas, P_update, tip_update, angle_update


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


def export_training_csv(path_csv: str, Strctr: "StructureClass", Sprvsr: "SupervisorClass", T: Optional[int] = None,
                        State_meas: Optional["StateClass"] = None, State_update: Optional["StateClass"] = None) -> None:
    """
    Export training outputs to a CSV file.

    Parameters
    ----------
    path_csv : str, output CSV file path.
    Strctr : StructureClass, for (`hinges`) and (`shims`).
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
    header = ["t"]

    # full internal state arrays
    if State_meas is not None:
        header += ["final_pos_meas"]
    if State_update is not None:
        header += ["final_pos_update"]

    # keep update command if you still want it
    header += ["upd_x_tip", "upd_y_tip", "upd_tip_angle"]

    # losses
    loss_size = Sprvsr.loss_in_t.shape[1]
    header += [f"loss_{i}" for i in range(loss_size)]
    header += ["loss_MSE"]

    if State_meas is not None:  # measured forces
        header += ["Fx_meas", "Fy_meas"]
    header += ["Fx_des", "Fy_des"]  # desired forces
    header += ["Fx_update", "Fy_update"]  # tip update forces

    # whole buckle arrays
    if State_meas is not None:
        header += ["buckle_arr_meas"]
    if State_update is not None:
        header += ["buckle_arr_update"]

    # chain intersects with itself
    if State_update.intersection_times is not None:
        header += ["intersection_times"]

    # ------ write ------
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        for t in range(T):
            row = [t]

            # full measured state positions
            if State_meas is not None:
                pos_meas = State_meas.pos_arr_in_t[:, :, t] * Sprvsr.convert_pos
                row += [arr_to_json(pos_meas)]

            # full updated state positions
            if State_update is not None:
                pos_update = State_update.pos_arr_in_t[:, :, t] * Sprvsr.convert_pos
                row += [arr_to_json(pos_update)]

            # update command, kept as scalars
            row += [float(tip_pos_update_in_t[t, 0]), float(tip_pos_update_in_t[t, 1]), float(angle_update_in_t[t])]

            # losses
            row += [float(x) for x in Sprvsr.loss_in_t[t, :]]
            row += [float(Sprvsr.loss_MSE_in_t[t])]

            if State_meas is not None:  # measured forces
                row += [float(meas_Fx[t]), float(meas_Fy[t])]
            row += [float(des_Fx[t]), float(des_Fy[t])]  # desired forces
            row += [float(update_Fx[t]), float(update_Fy[t])]  # update forces

            # full buckle arrays
            if State_meas is not None:
                row += [arr_to_json(State_meas.buckle_in_t[:, :, t])]
            if State_update is not None:
                row += [arr_to_json(State_update.buckle_in_t[:, :, t])]

            # chain intersects with itself
            if State_update.intersection_times is not None:
                row += [int(State_update.intersection_times[t]) if t < len(State_update.intersection_times) else int(0)]

            w.writerow(row)


def export_training_npz(path_npz: str, **arrays):
    """
    Save big arrays (pos/angles/buckles) in one compressed file.
    """
    path_npz = Path(path_npz)
    path_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path_npz, **arrays)


# ---------------------------------------------------------------
# Post-processing files
# ---------------------------------------------------------------
def loss_from_filename(file: Path):
    return float(re.search(r"final_loss_(.*?)_init_", file.stem).group(1))


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
def buckle_transitions(folder: str | Path, only_init_and_final_buckles: bool = False, omit_inverted: bool = False):
    """
    Go over all final_loss_*.csv files and extract directed buckle transitions.

    Parameters
    ----------
    folder                      : path, all csv run files, from every init to every desired
    only_init_and_final_buckles : bool, True = transition is only from initial to final (not necessarily the desired)
                                  desired transition colored Cyan, undesired colored purple
    omit_inverted               : bool, True = do not account for  "_inverted.csv" output files

    Returns
    -------
    transitions          : Counter[(src, dst)] = number of times observed across all files
    per_file_transitions : dict[file_name, list[(src, dst)]]
    per_file_loss        : dict[file_name, float]
    edge_zero_loss_count : Counter[(src, dst)] = number of zero-loss files on this edge
    missing_edges        : ???
    """
    folder = Path(folder)
    transitions, per_file_transitions, per_file_loss, edge_zero_loss_count = helpers_builders.build_transition_counts(folder,
                                                                                                                      only_init_and_final_buckles=only_init_and_final_buckles,
                                                                                                                      omit_inverted=omit_inverted)

    observed_edges = set(transitions.keys())
    missing_edges = [edge for edge in helpers_builders.all_possible_transitions(4) if
                     edge not in observed_edges and helpers_builders.hamming_distance_int(*edge) == 1]

    print(f"Found {len(per_file_transitions)} files")
    print(f"Found {sum(transitions.values())} total transitions")
    print(f"Found {len(transitions)} unique directed transitions\n")

    print("Top transitions:")
    for (a, b), c in transitions.most_common(20):
        print(f"{helpers_builders.index_to_buckle(a)} -> {helpers_builders.index_to_buckle(b)}: {c}")

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
    return any(p.is_file() and not p.name.startswith("log_") for p in Path(prev_dir).glob(pattern))


# -----------------------------
# File helpers
# -----------------------------
def correct_buckle_string(buckle_arr: NDArray):
    buckle = copy.copy(buckle_arr)
    buckle[buckle_arr == -1] = 0
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