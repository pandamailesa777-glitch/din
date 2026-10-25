"""Self-contained, versioned calculation inputs; separate from result exports."""
from dataclasses import asdict
import json
from pathlib import Path

from .inputs import InputError, Load, Parameters


def save_case(path, parameters, load, title="Вертикальное движение"):
    load.validate_coverage(parameters)
    data = {"schema": "vertical-dynamics-case/1", "title": title,
            "units": "SI: kg, m, s, N", "parameters": asdict(parameters),
            "load": {"time_s": load.time.tolist(), "force_N": load.force.tolist()}}
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_case(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict) or data.get("schema") != "vertical-dynamics-case/1":
            raise InputError("Неизвестный формат варианта расчёта.")
        if data.get("units") != "SI: kg, m, s, N":
            raise InputError("Вариант должен явно задавать систему СИ: kg, m, s, N.")
        if not isinstance(data.get("title"), str) or len(data["title"]) > 200:
            raise InputError("Название расчёта: строка длиной до 200 символов.")
        p = Parameters(**data["parameters"])
        load = Load(data["load"]["time_s"], data["load"]["force_N"])
        load.validate_coverage(p)
        return p, load, data["title"]
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise InputError(f"Не удалось загрузить вариант: {exc}") from exc
