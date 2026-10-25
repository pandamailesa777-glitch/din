from dataclasses import replace
import json

import numpy as np
import pytest
from pypdf import PdfReader

from vertical_dynamics import InputError, Load, load_force, simulate
from vertical_dynamics.case import load_case, save_case
from vertical_dynamics.results import format_value, summary, time_history
from vertical_dynamics.report import conclusion, create_report
from test_core import parameters, constant


@pytest.mark.parametrize("method", ["RK45", "Euler"])
@pytest.mark.parametrize("force", [0, 99, 100, -100])
def test_both_methods_remain_at_rest(method, force):
    result = simulate(parameters(method=method), constant(force))
    data = time_history(result)
    assert not result.detached
    assert (data.z_m == 0).all() and (data.vz_m_s == 0).all()
    assert (data.reaction_N >= 0).all()


def test_euler_first_order_convergence_height_and_impact():
    # m=10 kg, g=10, F=200 N for 1 s: zmax=10, t_hit=2+sqrt(2).
    exact_time = 2 + np.sqrt(2)
    errors = []
    for h in (.08, .04, .02, .01):
        result = simulate(parameters(method="Euler", euler_step=h), constant(200))
        values = summary(result).set_index("key").value
        error = abs(result.touchdown - exact_time)
        errors.append(error)
        assert abs(values.max_height - 10) < 1.0
        assert values.impact_velocity < 0
        history = time_history(result)
        assert history.z_m.min() >= 0
        assert history.reaction_N.dropna().min() >= 0
    assert all(b < .65*a for a, b in zip(errors, errors[1:]))
    rk = simulate(parameters(), constant(200))
    assert abs(rk.touchdown - exact_time) < 1e-9
    assert errors[-1] < .02


def test_euler_ramp_converges_and_localizes_lift():
    load = Load([0, .5, 1], [0, 100, 200])
    rk = simulate(parameters(), load)
    errors = []
    for h in (.02, .01, .005):
        result = simulate(parameters(method="Euler", euler_step=h), load)
        assert result.lift_off == .5
        errors.append(abs(result.touchdown-rk.touchdown))
    assert errors[2] < errors[1] < errors[0]


@pytest.mark.parametrize("method", ["RK45", "Euler"])
def test_initial_flight_and_unfinished_flight(method):
    fall = simulate(parameters(method=method, initial_height=5), constant(0))
    assert fall.initially_airborne and fall.lift_off is None
    assert fall.touchdown is not None
    unfinished = simulate(parameters(method=method, end_time=1), constant(200))
    assert unfinished.touchdown is None
    assert "не зарегистрировано" in conclusion(unfinished)
    values = summary(unfinished).set_index("key").value
    assert values.observed_flight_duration == 1
    assert values.flight_duration is None
    assert format_value(values.touchdown_time) == "Не определено"


def test_case_roundtrip_all_settings_and_force(tmp_path):
    path = tmp_path / "variant.json"
    p = parameters(method="Euler", euler_step=.03)
    load = Load([0, .5, 1], [-100, 200, 0])
    save_case(path, p, load, "Проверка №1")
    restored, restored_load, title = load_case(path)
    assert restored == p and title == "Проверка №1"
    np.testing.assert_array_equal(restored_load.force, load.force)
    np.testing.assert_array_equal(restored_load.time, load.time)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["units"] = "mm, kgf"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(InputError, match="СИ"):
        load_case(path)


def test_kgf_explicit_conversion(tmp_path):
    path = tmp_path / "force.csv"
    path.write_text("time[ms],Fz[kgf]\n0,1\n1000,-2\n", encoding="utf-8")
    load = load_force(path)
    np.testing.assert_allclose(load.force, [9.80665, -19.6133])
    np.testing.assert_array_equal(load.time, [0, 1])


@pytest.mark.parametrize("changes", [{"method": "invalid"}, {"euler_step": 0}, {"method": "Euler", "euler_step": 1e-9}])
def test_invalid_solver_settings(changes):
    with pytest.raises(InputError):
        parameters(**changes)


def test_report_cyrillic_sections_and_results(tmp_path):
    result = simulate(parameters(), constant(200))
    path = create_report(result, tmp_path / "report.pdf", "Контрольный расчёт")
    reader = PdfReader(path)
    text = " ".join(" ".join(page.extract_text() for page in reader.pages).split())
    assert len(reader.pages) >= 4
    for expected in ("Контрольный расчёт", "Исходные данные", "Математическая модель", "Основные результаты", "Ограничения", "Графики", "200", "не подтверждает безопасность", "RK45"):
        assert expected in text
