from __future__ import annotations

import os
import copy
import traceback
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from config import CFG
from StructureClass import StructureClass
from VariablesClass import VariablesClass
from SupervisorClass import SupervisorClass
from StateClass import StateClass
from EquilibriumClass import EquilibriumClass

import numerical_experiments, plot_funcs, file_funcs


def _train_one_pair(init_buckle, desired_buckle, invert_updates=False):
    Strctr = StructureClass(CFG, update_scheme=CFG.Train.update_scheme)
    Variabs = VariablesClass(Strctr, CFG)
    Sprvsr = SupervisorClass(Strctr, CFG, supress_prints=True)

    Sprvsr, State_meas, State_des, State_update, Eq_meas, Eq_des, t = numerical_experiments.train(Strctr, Variabs,
                                                                                                  CFG, init_buckle,
                                                                                                  desired_buckle,
                                                                                                  invert_updates=invert_updates)
    return Strctr, Variabs, Sprvsr, State_meas, State_des, State_update, Eq_meas, Eq_des, t


def run_one_job(job):
    k = int(job["k"])
    l = int(job["l"])
    init_buckle_tup = tuple(job["init_buckle_tup"])
    desired_buckle_tup = tuple(job["desired_buckle_tup"])
    run_dir = Path(job["run_dir"])
    save_gifs = bool(job["save_gifs"])
    save_pngs = bool(job["save_pngs"])
    save_csvs = bool(job["save_csvs"])

    run_dir.mkdir(parents=True, exist_ok=True)

    init_buckle = np.array(init_buckle_tup, dtype=np.int32).reshape(4, 1)
    desired_buckle = np.array(desired_buckle_tup, dtype=np.int32).reshape(4, 1)

    init_buckle_str = file_funcs.correct_buckle_string(init_buckle)
    desired_buckle_str = file_funcs.correct_buckle_string(desired_buckle)

    log_path = run_dir / f"log_init_{init_buckle_str}_desired_{desired_buckle_str}.txt"

    try:
        with open(log_path, "w", encoding="utf-8") as log_f, redirect_stdout(log_f), redirect_stderr(log_f):
            # Prevent per-step interactive plotting inside worker processes.
            plot_funcs.plot_arm = lambda *args, **kwargs: None

            Strctr, Variabs, Sprvsr, State_meas, State_des, State_update, Eq_meas, Eq_des, t = _train_one_pair(init_buckle=init_buckle,
                                                                                                               desired_buckle=desired_buckle,
                                                                                                               invert_updates=False)

            F_meas_in_t = np.array([State_meas.Fx_in_t, State_meas.Fy_in_t])
            F_des_in_t = np.array([State_des.Fx_in_t, State_des.Fy_in_t])

            gif_path = None
            png_path = None
            csv_path = None

            if save_gifs:
                pos_in_t_update = np.moveaxis(State_update.pos_arr_in_t, 2, 0)
                buckle_in_t = np.moveaxis(State_meas.buckle_in_t, 2, 0)
                final_frame = min(t + 3, pos_in_t_update.shape[0])
                gif_path = str(run_dir / f"gif_init_{init_buckle_str}_desired_{desired_buckle_str}.gif")
                plot_funcs.animate_arm_w_arcs(
                    pos_in_t_update[1:final_frame, :, :],
                    Strctr.L,
                    frames=max(1, final_frame - 1),
                    interval_ms=400,
                    save_path=str(gif_path),
                    fps=2,
                    buckle_traj=buckle_in_t[1:final_frame, :, :],
                )
                plt.close("all")

            if save_pngs:
                png_path = str(run_dir / f"final_loss_{Sprvsr.loss_MSE_in_t[t]:.6g}_init_{init_buckle_str}_desired_{desired_buckle_str}.png")
                plot_funcs.loss_and_buckle_in_t(Sprvsr.tip_pos_in_t, Sprvsr.tip_angle_in_t, Sprvsr.loss_in_t, State_update.buckle_in_t, 
                                                F_meas_in_t, F_des_in_t, Sprvsr.tip_pos_update_in_t, Sprvsr.tip_angle_update_in_t, 
                                                start=0, end=t, save_path=png_path)
                plt.close("all")

            if save_csvs:
                csv_path = str(run_dir / f"final_loss_{Sprvsr.loss_MSE_in_t[t]:.6g}_init_{init_buckle_str}_desired_{desired_buckle_str}.csv")
                file_funcs.export_training_csv(
                    str(csv_path),
                    Strctr,
                    Sprvsr,
                    T=t + 1,
                    State_meas=State_meas,
                    State_update=State_update,
                )

            if Sprvsr.loss_MSE > 10**(-6):
                print('failed to train with Sprvsr.invert_delta_tip=', Sprvsr.invert_delta_tip)
                Strctr, Variabs, Sprvsr, State_meas, State_des, State_update, Eq_meas, Eq_des, t = _train_one_pair(init_buckle=init_buckle,
                                                                                                                   desired_buckle=desired_buckle,
                                                                                                                   invert_updates=True)

                F_meas_in_t = np.array([State_meas.Fx_in_t, State_meas.Fy_in_t])
                F_des_in_t = np.array([State_des.Fx_in_t, State_des.Fy_in_t])

                gif_path = None
                png_path = None
                csv_path = None

                if save_gifs:
                    pos_in_t_update = np.moveaxis(State_update.pos_arr_in_t, 2, 0)
                    buckle_in_t = np.moveaxis(State_meas.buckle_in_t, 2, 0)
                    final_frame = min(t + 3, pos_in_t_update.shape[0])
                    gif_path = str(run_dir / f"gif_init_{init_buckle_str}_desired_{desired_buckle_str}_inverted.gif")
                    plot_funcs.animate_arm_w_arcs(
                        pos_in_t_update[1:final_frame, :, :],
                        Strctr.L,
                        frames=max(1, final_frame - 1),
                        interval_ms=400,
                        save_path=str(gif_path),
                        fps=2,
                        buckle_traj=buckle_in_t[1:final_frame, :, :],
                    )
                    plt.close("all")

                if save_pngs:
                    png_path = str(run_dir / f"final_loss_{Sprvsr.loss_MSE_in_t[t]:.6g}_init_{init_buckle_str}_desired_{desired_buckle_str}_inverted.png")
                    plot_funcs.loss_and_buckle_in_t(Sprvsr.tip_pos_in_t, Sprvsr.tip_angle_in_t, Sprvsr.loss_in_t, State_update.buckle_in_t, 
                                                    F_meas_in_t, F_des_in_t, Sprvsr.tip_pos_update_in_t, Sprvsr.tip_angle_update_in_t, 
                                                    start=0, end=t, save_path=png_path)
                    plt.close("all")

                if save_csvs:
                    csv_path = str(run_dir / f"final_loss_{Sprvsr.loss_MSE_in_t[t]:.6g}_init_{init_buckle_str}_desired_{desired_buckle_str}_inverted.csv")
                    file_funcs.export_training_csv(
                        str(csv_path),
                        Strctr,
                        Sprvsr,
                        T=t + 1,
                        State_meas=State_meas,
                        State_update=State_update,
                    )

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
        }

    except Exception:
        with open(log_path, "a", encoding="utf-8") as log_f:
            log_f.write("\n\n=== EXCEPTION ===\n")
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
        }
