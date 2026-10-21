"""Strict input parsing and explicit dimension conversion to SI."""
from dataclasses import dataclass
from decimal import Decimal
import json
from pathlib import Path

import numpy as np
import pandas as pd


class InputError(ValueError):
    pass


UNITS = {
    "mass": {"kg": 1.0, "g": 0.001, "t": 1000.0},
    "acceleration": {"m/s^2": 1.0, "mm/s^2": 0.001},
    "velocity": {"m/s": 1.0, "mm/s": 0.001},
    "length": {"m": 1.0, "mm": 0.001, "cm": 0.01},
    "time": {"s": 1.0, "ms": 0.001},
    "force": {"N": 1.0, "kN": 1000.0},
}


def number(value, name):
    if isinstance(value, (bool, str)):
        raise InputError(f"{name}: ожидается конечное число.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputError(f"{name}: ожидается конечное число.") from exc
    if not np.isfinite(result):
        raise InputError(f"{name}: значение должно быть конечным.")
    return result


def quantity(data, key, dimension):
    item = data.get(key)
    if not isinstance(item, dict) or set(item) != {"value", "unit"}:
        raise InputError(f"{key}: обязательны ровно поля value и unit.")
    unit = item["unit"]
    if not isinstance(unit, str) or unit not in UNITS[dimension]:
        raise InputError(f"{key}: недопустимая единица {unit!r}; допустимы {list(UNITS[dimension])}.")
    # Decimal scaling avoids 700 ms becoming 0.7000000000000001 s and
    # incorrectly failing coverage against a CSV endpoint of 0.7 s.
    value = number(item["value"], key)
    return float(Decimal(str(value)) * Decimal(str(UNITS[dimension][unit])))


@dataclass(frozen=True)
class Parameters:
    mass: float
    gravity: float
    initial_velocity: float
    initial_height: float
    force_start: float
    force_end: float
    end_time: float
    max_step: float
    output_step: float
    rtol: float
    atol_z: float
    atol_v: float

    def __post_init__(self):
        for key, value in vars(self).items():
            number(value, key)
        for key in ("mass", "gravity", "end_time", "max_step", "output_step", "atol_z", "atol_v"):
            if getattr(self, key) <= 0:
                raise InputError(f"{key}: требуется положительное значение.")
        if not 1e-12 <= self.rtol < 1:
            raise InputError("rtol должен лежать в диапазоне [1e-12, 1).")
        if not 0 <= self.force_start < self.force_end <= self.end_time:
            raise InputError("Требуется 0 ≤ t0 < t1 ≤ tmax.")
        if self.initial_height < 0:
            raise InputError("z0 не может быть ниже основания.")
        if self.initial_height == 0 and self.initial_velocity < 0:
            raise InputError("z0 = 0 и Vz0 < 0 задают начальный удар; модель его не описывает.")
        if self.end_time / self.output_step > 1_000_000:
            raise InputError("Слишком много строк вывода: увеличьте output_step (предел 1 000 000).")


def load_config(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise InputError(f"Не удалось прочитать JSON: {exc}") from exc
    if not isinstance(data, dict) or set(data) != {"parameters", "solver"}:
        raise InputError("JSON должен содержать ровно разделы parameters и solver.")
    p, s = data["parameters"], data["solver"]
    if not isinstance(p, dict) or set(p) != {"m", "g", "Vz0", "z0", "t0", "t1", "tmax"}:
        raise InputError("parameters: обязательны m, g, Vz0, z0, t0, t1, tmax; лишние поля запрещены.")
    if not isinstance(s, dict) or set(s) != {"max_step", "output_step", "rtol", "atol_z", "atol_v"}:
        raise InputError("solver: обязательны max_step, output_step, rtol, atol_z, atol_v.")
    return Parameters(
        quantity(p, "m", "mass"), quantity(p, "g", "acceleration"),
        quantity(p, "Vz0", "velocity"), quantity(p, "z0", "length"),
        quantity(p, "t0", "time"), quantity(p, "t1", "time"), quantity(p, "tmax", "time"),
        quantity(s, "max_step", "time"), quantity(s, "output_step", "time"),
        number(s["rtol"], "rtol"), quantity(s, "atol_z", "length"), quantity(s, "atol_v", "velocity"),
    )


@dataclass(frozen=True)
class Load:
    time: np.ndarray
    force: np.ndarray

    def __post_init__(self):
        try:
            t, f = np.array(self.time, dtype=float, copy=True), np.array(self.force, dtype=float, copy=True)
        except (ValueError, TypeError) as exc:
            raise InputError("Нагрузка должна содержать числовые массивы.") from exc
        if t.ndim != 1 or f.ndim != 1 or len(t) != len(f) or len(t) < 2:
            raise InputError("Нужны как минимум две пары времени и силы.")
        if not np.isfinite(t).all() or not np.isfinite(f).all() or not np.all(np.diff(t) > 0):
            raise InputError("CSV: значения должны быть конечными, время — строго возрастающим без повторов.")
        if t[0] < 0:
            raise InputError("CSV: время не может быть отрицательным.")
        t.setflags(write=False)
        f.setflags(write=False)
        object.__setattr__(self, "time", t)
        object.__setattr__(self, "force", f)

    def validate_coverage(self, p):
        if self.time[0] > p.force_start or self.time[-1] < p.force_end:
            raise InputError("CSV должен покрывать весь интервал [t0, t1]; экстраполяция запрещена.")

    def value(self, t, p):
        t = np.asarray(t)
        return np.where((t >= p.force_start) & (t < p.force_end), np.interp(t, self.time, self.force), 0.0)

    def linear_piece(self, left, right, p):
        """Force at left and slope; interval must not contain an interior knot."""
        mid = (left + right) / 2
        if not p.force_start <= mid < p.force_end:
            return 0.0, 0.0
        index = int(np.clip(np.searchsorted(self.time, mid) - 1, 0, len(self.time) - 2))
        slope = (self.force[index + 1] - self.force[index]) / (self.time[index + 1] - self.time[index])
        return float(self.force[index] + slope * (left - self.time[index])), float(slope)


def load_force(path):
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise InputError(f"Не удалось прочитать CSV нагрузки: {exc}") from exc
    if len(frame.columns) != 2:
        raise InputError("CSV должен иметь ровно два столбца: time[единица], Fz[единица].")
    units = []
    for column, prefix, dimension in zip(frame.columns, ("time", "Fz"), ("time", "force")):
        valid = {f"{prefix}[{unit}]": factor for unit, factor in UNITS[dimension].items()}
        if column not in valid:
            raise InputError(f"Недопустимый столбец {column!r}; допустимы {list(valid)}.")
        units.append(valid[column])
    try:
        values = frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise InputError("CSV содержит нечисловые значения.") from exc
    converted = [np.array([float(Decimal(str(v)) * Decimal(str(factor))) for v in values[:, i]])
                 for i, factor in enumerate(units)]
    return Load(*converted)
