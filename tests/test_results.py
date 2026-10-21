from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from vertical_dynamics import simulate
from vertical_dynamics.plotting import make_figure, save_png
from vertical_dynamics.results import export_csv, sample, summary, time_history
from test_core import constant, parameters


def values(result):
    return summary(result).set_index("key").value


def test_extrema_do_not_depend_on_output_step():
    for step in (0.005, 0.9, 5):
        result = simulate(parameters(output_step=step), constant(200))
        s = values(result)
        assert s.max_height == pytest.approx(10)
        assert s.max_velocity == pytest.approx(10)
        assert s.max_speed == pytest.approx(10 * np.sqrt(2))
        assert s.max_abs_acceleration == pytest.approx(10)


def test_velocity_extremum_inside_ramp():
    from vertical_dynamics import Load
    result = simulate(parameters(initial_height=100, end_time=1), Load([0, 1], [200, 0]))
    assert values(result).max_velocity == pytest.approx(2.5)


def test_contact_rows_and_touchdown_semantics():
    p = parameters(force_start=0.2)
    result = simulate(p, constant(200))
    table = time_history(result)
    assert table.z_m.min() >= 0
    assert table.iloc[0].state == "contact"
    assert table.iloc[0].reaction_N == 100
    assert table.iloc[-1].state == "touchdown"
    assert table.iloc[-1].z_m == 0
    assert table.iloc[-1].vz_m_s < 0
    assert np.isnan(table.iloc[-1].reaction_N)
    assert table.time_s.is_monotonic_increasing and table.time_s.is_unique


def test_initial_airborne_max_signed_acceleration():
    result = simulate(parameters(initial_height=5), constant(0))
    s = values(result)
    assert s.max_acceleration == -10
    assert s.max_abs_acceleration == 10
    assert s.max_height == 5
    assert s.max_rise == 0
    assert s.lift_off_time is None


def test_export_and_png(tmp_path):
    result = simulate(parameters(), constant(200))
    paths = export_csv(result, tmp_path)
    table = pd.read_csv(paths[0])
    assert table.iloc[-1].time_s == pytest.approx(result.touchdown)
    assert len(pd.read_csv(paths[1])) == 13
    assert json.loads(paths[2].read_text(encoding="utf-8"))["parameters_SI"]["mass"] == 10
    path = tmp_path / "plots.png"
    save_png(result, path)
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    fig = make_figure(result)
    assert len(fig.axes) == 5
    assert sum(len(ax.lines) for ax in fig.axes) >= 15


def test_missing_events_are_empty_and_explained():
    result = simulate(parameters(), constant(0))
    report = summary(result).set_index("key")
    assert report.loc["touchdown_time", "value"] is None
    assert report.loc["touchdown_time", "note"]
    assert values(result).max_height == 0
    assert (time_history(result).az_m_s2 == 0).all()


def test_invalid_sampling_rejected():
    result = simulate(parameters(), constant(0))
    with pytest.raises(ValueError):
        sample(result, [-1, 0])
