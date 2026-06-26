from __future__ import annotations

import os
import copy
import traceback
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
from typing import Optional, TypedDict, Union
from datetime import datetime
from time import time

from config import CFG
from StructureClass import StructureClass
from VariablesClass import VariablesClass
from SupervisorClass import SupervisorClass
from StateClass import StateClass
from EquilibriumClass import EquilibriumClass

import numerical_experiments, plot_funcs, file_funcs

matplotlib.use("Agg", force=True)


def _local_log_time() -> str:
    """Return a readable local timestamp for worker log files."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class ParallelJob(TypedDict):
    """Input specification for one parallel buckle-training run."""

    k: int
    l: int
    init_buckle_tup: tuple[int, ...]
    desired_buckle_tup: tuple[int, ...]
    run_dir: Union[str, Path]
    save_gifs: bool
    save_pngs: bool
    save_csvs: bool
    pos_delta_mode: str


class ParallelJobResult(TypedDict):
    """Result summary returned by one parallel buckle-training run."""

    ok: bool
    k: int
    l: int
    init_buckle_tup: tuple[int, ...]
    desired_buckle_tup: tuple[int, ...]
    loss: float
    gif_path: Optional[str]
    png_path: Optional[str]
    csv_path: Optional[str]
    log_path: str
    pos_delta_mode: str


def _train_one_pair(init_buckle: np.ndarray, desired_buckle: np.ndarray,
                    invert_updates: bool = False,
                    pos_delta_mode: Optional[str] = None) -> tuple[StructureClass, VariablesClass, SupervisorClass,
                                                                   StateClass, StateClass, StateClass, EquilibriumClass,
                                                                   EquilibriumClass, int]:
    cfg = CFG
    if pos_delta_mode is not None:
        cfg = replace(CFG, Train=replace(CFG.Train, pos_delta_mode=pos_delta_mode))

    Strctr: "StructureClass" = StructureClass(cfg, update_scheme=cfg.Train.update_scheme)
    Variabs: "VariablesClass" = VariablesClass(Strctr, cfg)
    Sprvsr: "SupervisorClass" = SupervisorClass(Strctr, cfg, supress_prints=True)

    Sprvsr, State_meas, State_des, State_update, Eq_meas, Eq_des, t = numerical_experiments.train(Strctr, Variabs,
                                                                                                  cfg, init_buckle,
                                                                                                  desired_buckle,
                                                                                                  invert_updates=invert_updates)
    return Strctr, Variabs, Sprvsr, State_meas, State_des, State_update, Eq_meas, Eq_des, t


def run_one_job(job: ParallelJob) -> ParallelJobResult:
    """
    Run one buckle-training and save requested output artifacts.

    Parameters
    ----------
    job
        Dictionary describing one training run. Required keys are:
        k, l               - Integer success_matrix indices for identifying result.
        init_buckle_tup    - Initial buckle pattern as a length-4 tuple of ``-1``/``+1`` values.
        desired_buckle_tup - Desired buckle pattern as a length-4 tuple of ``-1``/``+1`` values.
        run_dir            - Directory of log file & optional output files to write (created if not existing)
        save_gifs          - True = save HTML animation of update trajectory in training time.
        save_pngs          - True = save sizes during training time t as graphs in PNG.
        save_csvs          - True = export training trajectory and state history to CSV.

    Returns
    -------
    ParallelJobResult
        Dictionary summarizing the run:
        ok - True = run completed without raising exception.
        k, l - The input job indices, copied into the result.
        init_buckle_tup, desired_buckle_tup - The input buckle patterns, copied into the result.
        loss - Final scalar MSE-like loss from ``Sprvsr.loss_MSE`` on success. ``np.nan`` if run failed.
        gif_path, png_path, csv_path - Paths to generated output files, None if not requested or if run failed.
        log_path - Path to per-job log file. On failure, traceback is appended to this file.

    Notes
    -----
    Interactive per-step plotting is disabled inside the worker process by
    replacing ``plot_funcs.plot_arm`` with a no-op.
    """
    k = int(job["k"])
    l = int(job["l"])
    init_buckle_tup = tuple(job["init_buckle_tup"])
    desired_buckle_tup = tuple(job["desired_buckle_tup"])
    run_dir = Path(job["run_dir"])
    save_gifs = bool(job["save_gifs"])
    save_pngs = bool(job["save_pngs"])
    save_csvs = bool(job["save_csvs"])
    pos_delta_mode = str(job.get("pos_delta_mode", CFG.Train.pos_delta_mode))

    run_dir.mkdir(parents=True, exist_ok=True)

    H, S = CFG.Strctr.H, CFG.Strctr.S
    expected_buckle_size = H * S
    if len(init_buckle_tup) != expected_buckle_size or len(desired_buckle_tup) != expected_buckle_size:
        log_path = run_dir / f"log_invalid_job_k_{k}_l_{l}_{pos_delta_mode}.txt"
        with open(log_path, "w", encoding="utf-8") as log_f:
            log_f.write(
                f"Invalid buckle length for CFG.Strctr.H={H}, CFG.Strctr.S={S}. "
                f"Expected {expected_buckle_size}, got init={len(init_buckle_tup)}, "
                f"desired={len(desired_buckle_tup)}.\n"
                f"local start time={_local_log_time()}\n"
            )
        return {
            "ok": False,
            "k": k,
            "l": l,
            "init_buckle_tup": init_buckle_tup,
            "desired_buckle_tup": desired_buckle_tup,
            "loss": np.nan,
            "gif_path": None,
            "png_path": None,
            "csv_path": None,
            "log_path": str(log_path),
            "pos_delta_mode": pos_delta_mode,
        }

    init_buckle = np.asarray(init_buckle_tup, dtype=np.int32)
    desired_buckle = np.asarray(desired_buckle_tup, dtype=np.int32)

    init_buckle = init_buckle.reshape(H, S)
    desired_buckle = desired_buckle.reshape(H, S)

    init_buckle_str = file_funcs.correct_buckle_string(init_buckle)
    desired_buckle_str = file_funcs.correct_buckle_string(desired_buckle)

    log_path = run_dir / f"log_init_{init_buckle_str}_desired_{desired_buckle_str}.txt"

    try:
        with open(log_path, "w", encoding="utf-8") as log_f, redirect_stdout(log_f), redirect_stderr(log_f):
            job_t0 = time()
            print(f"local start time={_local_log_time()}")
            print(f"job k={k}, l={l}, init={init_buckle_str}, desired={desired_buckle_str}, pos_delta_mode={pos_delta_mode}")

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="A JAX array is being set as static!.*", category=UserWarning)
                warnings.filterwarnings("ignore", message="Using `field\\(init=False\\)` on `equinox\\.Module`.*", category=UserWarning)

                # Prevent per-step interactive plotting inside worker processes.
                plot_funcs.plot_arm = lambda *args, **kwargs: None

                Strctr, Variabs, Sprvsr, State_meas, State_des, State_update, Eq_meas, Eq_des, t = _train_one_pair(init_buckle=init_buckle,
                                                                                                                   desired_buckle=desired_buckle,
                                                                                                                   invert_updates=False,
                                                                                                                   pos_delta_mode=pos_delta_mode)

            F_meas_in_t = np.array([State_meas.Fx_in_t, State_meas.Fy_in_t])
            F_des_in_t = np.array([State_des.Fx_in_t, State_des.Fy_in_t])

            gif_path = None
            png_path = None
            csv_path = None

            if save_gifs:
                pos_in_t_update = np.moveaxis(State_update.pos_arr_in_t, 2, 0)
                buckle_in_t = np.moveaxis(State_meas.buckle_in_t, 2, 0)
                final_frame = min(t + 3, pos_in_t_update.shape[0])
                # gif_path = str(run_dir / f"gif_init_{init_buckle_str}_desired_{desired_buckle_str}.gif")
                gif_path = str(run_dir / f"gif_init_{init_buckle_str}_desired_{desired_buckle_str}.html")
                plot_funcs.animate_arm_w_arcs(pos_in_t_update[1:final_frame, :, :], Strctr.L, Fx=State_update.Fx_in_t,
                                              Fy=State_update.Fy_in_t, frames=max(1, final_frame - 1), interval_ms=400,
                                              save_path=str(gif_path), fps=2,
                                              buckle_traj=buckle_in_t[1:final_frame, :, :])
                plt.close("all")

            if save_pngs:
                png_path = str(run_dir / f"final_loss_{Sprvsr.loss_MSE_in_t[t]:.6g}_init_{init_buckle_str}_desired_{desired_buckle_str}.png")
                plot_funcs.loss_and_buckle_in_t(Sprvsr.tip_pos_in_t, Sprvsr.tip_angle_in_t, Sprvsr.loss_in_t, State_update.buckle_in_t, 
                                                F_meas_in_t, F_des_in_t, Sprvsr.tip_pos_update_in_t, Sprvsr.tip_angle_update_in_t, 
                                                start=0, end=t, save_path=png_path)
                plt.close("all")

            if save_csvs:
                suffix1 = "_intersect" if State_update.self_intersection else ""  # append str if chain intersects
                suffix2 = "_flip_chain" if Sprvsr.symmetrical_state else ""    # append if chain is symmetrical to des
                csv_path = str(run_dir / f"final_loss_{Sprvsr.loss_MSE_in_t[t]:.6g}_init_{init_buckle_str}_desired_{desired_buckle_str}{suffix1}{suffix2}.csv")
                file_funcs.export_training_csv(str(csv_path), Strctr, Sprvsr, T=t + 1, State_meas=State_meas, State_update=State_update)

            print(f"local end time={_local_log_time()}")
            print(f"elapsed seconds={time() - job_t0:.2f}")
            print(f"final t={t}, loss={float(Sprvsr.loss_MSE):.6g}")

        return {
            "ok": True,
            "k": k,
            "l": l,
            "init_buckle_tup": init_buckle_tup,
            "desired_buckle_tup": desired_buckle_tup,
            "loss": float(Sprvsr.loss_MSE),
            "gif_path": None if gif_path is None else str(gif_path),
            "png_path": None if png_path is None else str(png_path),
            "csv_path": None if csv_path is None else str(csv_path),
            "log_path": str(log_path),
            "pos_delta_mode": pos_delta_mode,
        }

    except Exception:
        with open(log_path, "a", encoding="utf-8") as log_f:
            log_f.write(f"\n\n=== EXCEPTION at {_local_log_time()} ===\n")
            log_f.write(traceback.format_exc())
        return {
            "ok": False,
            "k": k,
            "l": l,
            "init_buckle_tup": init_buckle_tup,
            "desired_buckle_tup": desired_buckle_tup,
            "loss": np.nan,
            "gif_path": None,
            "png_path": None,
            "csv_path": None,
            "log_path": str(log_path),
            "pos_delta_mode": pos_delta_mode,
        }
