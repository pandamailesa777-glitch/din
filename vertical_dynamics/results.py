"""Dense-output sampling, extrema independent of output step, and SI export."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sample(result, times):
    p = result.parameters
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or not np.isfinite(t).all() or np.any(t < 0) or np.any(t > result.stop_time):
        raise ValueError("Время выборки должно находиться внутри рассчитанного интервала.")
    z = np.zeros_like(t)
    v = np.zeros_like(t)
    state = np.full(t.shape, "contact", dtype=object)
    for piece in result.pieces:
        mask = (t >= piece.start) & (t <= piece.end)
        if mask.any():
            z[mask], v[mask] = piece.solution(t[mask])
            state[mask] = "flight"
    tolerance = 100 * p.atol_z + 100 * np.finfo(float).eps * max(1, p.initial_height)
    if np.any(z < -tolerance):
        raise RuntimeError("Решатель обнаружил проникновение ниже основания; уменьшите шаг и допуски.")
    z = np.maximum(z, 0)
    force = np.asarray(result.load.value(t, p), dtype=float)
    a = np.where(state == "flight", force / p.mass - p.gravity, 0)
    reaction = np.where(state == "contact", np.maximum(p.mass * p.gravity - force, 0), 0.0)
    if result.touchdown is not None:
        mask = t == result.touchdown
        state[mask] = "touchdown"
        z[mask] = 0
        # Last row is pre-impact kinematics. No impact reaction is inferred.
        a[mask] = result.pieces[-1].acceleration(result.touchdown)
        force[mask] = p.mass * (a[mask] + p.gravity)
        reaction[mask] = np.nan
    return pd.DataFrame({"time_s": t, "z_m": z, "vz_m_s": v, "az_m_s2": a,
                         "Fz_N": force, "reaction_N": reaction, "state": state,
                         "contact": state != "flight"})


def time_history(result):
    count = int(np.floor(result.stop_time / result.parameters.output_step))
    grid = np.arange(count + 1) * result.parameters.output_step
    times = np.unique(np.r_[grid[grid <= result.stop_time], result.critical_times, result.stop_time])
    return sample(result, times)


def summary(result):
    p = result.parameters
    critical = sample(result, result.critical_times)
    accelerations = list(critical.az_m_s2)
    for piece in result.pieces:
        accelerations.extend([piece.acceleration(piece.start), piece.acceleration(piece.end)])
    height = float(critical.z_m.max())
    impact = None if result.touchdown is None else float(result.pieces[-1].solution(result.touchdown)[1])
    start = 0.0 if result.initially_airborne else result.lift_off
    duration = None if result.touchdown is None else result.touchdown - start
    rows = [
        ("detached", "Произошёл ли полный отрыв / отсутствие контакта", "Да" if result.detached else "Нет", ""),
        ("initially_airborne", "Начальное положение над основанием", "Да" if result.initially_airborne else "Нет", ""),
        ("max_height", "Максимальная высота над основанием", height, "м"),
        ("max_rise", "Максимальный подъём относительно z0", height - p.initial_height, "м"),
        ("max_velocity", "Максимальная вертикальная скорость (со знаком)", float(critical.vz_m_s.max()), "м/с"),
        ("max_speed", "Максимальный модуль вертикальной скорости", float(critical.vz_m_s.abs().max()), "м/с"),
        ("max_acceleration", "Максимальное ускорение (со знаком, без удара)", float(np.max(accelerations)), "м/с²"),
        ("max_abs_acceleration", "Максимальный модуль ускорения (без удара)", float(np.max(np.abs(accelerations))), "м/с²"),
        ("lift_off_time", "Момент отрыва", result.lift_off, "с"),
        ("touchdown_time", "Момент первого касания", result.touchdown, "с"),
        ("flight_duration", "Время полёта в расчётном интервале до касания", duration, "с"),
        ("impact_velocity", "Скорость непосредственно перед касанием", impact, "м/с"),
        ("stop_time", "Время окончания расчёта", result.stop_time, "с"),
    ]
    frame = pd.DataFrame(rows, columns=["key", "parameter", "value", "unit"])
    frame["note"] = ""
    if result.initially_airborne:
        frame.loc[frame.key == "lift_off_time", "note"] = "Отрыв до начала расчёта; время неизвестно"
    elif result.lift_off is None:
        frame.loc[frame.key == "lift_off_time", "note"] = "Отрыва не было"
    if result.touchdown is None:
        note = "Касание не наступило до tmax" if result.detached else "Конструкция оставалась на основании"
        frame.loc[frame.key.isin(["touchdown_time", "flight_duration", "impact_velocity"]), "note"] = note
    return frame


def format_value(value):
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.9g}" if isinstance(value, (float, int)) else str(value)


def export_csv(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / "time_history.csv", directory / "summary.csv", directory / "run_metadata.json"]
    time_history(result).to_csv(paths[0], index=False, encoding="utf-8-sig", float_format="%.15g")
    summary(result).to_csv(paths[1], index=False, encoding="utf-8-sig")
    metadata = {"model": "vertical rigid body; unilateral contact; stop at first touchdown",
                "parameters_SI": asdict(result.parameters),
                "load_time_s": result.load.time.tolist(), "load_force_N": result.load.force.tolist(),
                "force_interpolation": "piecewise linear on [t0,t1), zero outside",
                "touchdown_row": "pre-impact velocity, acceleration and force; impact reaction unknown"}
    paths[2].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths
