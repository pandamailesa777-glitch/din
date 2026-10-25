"""Reproducible analytical-reference report, including step refinement."""
import json
from pathlib import Path
import platform
import sys
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
import numpy as np
import pandas
import scipy

from vertical_dynamics import Load, Parameters, load_config, load_force, simulate
from vertical_dynamics.results import summary


def main():
    rows = []
    expected_time = 2 + np.sqrt(2)
    for step in (0.4, 0.2, 0.1, 0.05):
        p = Parameters(10, 10, 0, 0, 0, 1, 5, step, 0.02, 1e-9, 1e-11, 1e-11)
        result = simulate(p, Load([0, 1], [200, 200]))
        metrics = summary(result).set_index("key").value
        row = {"max_step_s": step, "touchdown_s": result.touchdown,
               "touchdown_error_s": abs(result.touchdown - expected_time),
               "height_error_m": abs(metrics.max_height - 10),
               "impact_velocity_error_m_s": abs(metrics.impact_velocity + 10 * np.sqrt(2))}
        assert max(row[key] for key in row if "error" in key) < 1e-10
        rows.append(row)
    example = simulate(load_config(ROOT / "examples/config.json"), load_force(ROOT / "examples/force.csv"))
    euler_rows = []
    for step in (.08, .04, .02, .01, .005):
        euler = simulate(replace(p, method="Euler", euler_step=step), Load([0, 1], [200, 200]))
        metrics = summary(euler).set_index("key").value
        euler_rows.append({"step_s": step, "touchdown_s": euler.touchdown,
                           "touchdown_error_s": abs(euler.touchdown-expected_time),
                           "height_error_m": abs(metrics.max_height-10),
                           "impact_velocity_error_m_s": abs(metrics.impact_velocity+10*np.sqrt(2))})
    assert all(b["touchdown_error_s"] < a["touchdown_error_s"] for a, b in zip(euler_rows, euler_rows[1:]))
    report = {"environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "scipy": scipy.__version__, "pandas": pandas.__version__, "matplotlib": matplotlib.__version__},
              "reference": {"m_kg": 10, "g_m_s2": 10, "force_N": 200, "pulse_duration_s": 1,
                            "touchdown_s": expected_time, "max_height_m": 10, "impact_velocity_m_s": -10 * np.sqrt(2)},
              "step_refinement": rows,
              "euler_step_refinement": euler_rows,
              "example": {row.key: row.value for row in summary(example).itertuples(index=False)}}
    path = ROOT / "docs/validation_results.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
