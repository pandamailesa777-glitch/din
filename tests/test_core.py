from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from vertical_dynamics import InputError, Load, Parameters, load_config, load_force, simulate
from vertical_dynamics.model import contact_reaction


def parameters(**changes):
    p = Parameters(10, 10, 0, 0, 0, 1, 5, 0.1, 0.02, 1e-9, 1e-11, 1e-11)
    return replace(p, **changes)


def constant(value):
    return Load([0, 1], [value, value])


def test_rest_and_non_tensile_reaction():
    for force in (0, 99, 100, -1000):
        result = simulate(parameters(), constant(force))
        assert not result.detached
        assert result.touchdown is None
        assert result.pieces == []
    assert contact_reaction(200, 10, 10) == 0
    assert contact_reaction(0, 10, 10) == 100


def test_ballistic_fall_and_no_penetration():
    r = simulate(parameters(initial_height=5), constant(0))
    assert r.initially_airborne and r.lift_off is None
    assert r.touchdown == pytest.approx(1, abs=1e-10)
    assert r.pieces[-1].solution(r.touchdown)[1] == pytest.approx(-10)
    for piece in r.pieces:
        t = np.linspace(piece.start, piece.end, 101)
        z, v = piece.solution(t)
        np.testing.assert_allclose(z, 5 - 5 * t**2, atol=1e-10)
        np.testing.assert_allclose(v, -10 * t, atol=1e-10)
        assert z.min() >= -1e-10


def test_initial_upward_velocity():
    r = simulate(parameters(initial_velocity=10), constant(0))
    assert r.lift_off == 0
    assert r.touchdown == pytest.approx(2, abs=1e-10)
    assert r.pieces[-1].solution(2)[1] == pytest.approx(-10)


def test_rectangular_pulse_exact_solution():
    r = simulate(parameters(), constant(200))
    assert r.lift_off == 0
    expected = 2 + np.sqrt(2)
    assert r.touchdown == pytest.approx(expected, abs=1e-10)
    assert r.pieces[-1].solution(r.touchdown)[1] == pytest.approx(-10 * np.sqrt(2))


def test_ramp_lift_threshold():
    r = simulate(parameters(), Load([0, 1], [0, 200]))
    assert r.lift_off == pytest.approx(0.5)
    z, v = r.pieces[0].solution(1)
    assert z == pytest.approx(20 * 0.5**3 / 6)
    assert v == pytest.approx(20 * 0.5**2 / 2)
    assert r.touchdown > 1


def test_touching_threshold_without_exceeding_does_not_lift():
    r = simulate(parameters(), Load([0, 0.5, 1], [0, 100, 0]))
    assert not r.detached


def test_delayed_discontinuity_and_short_pulse():
    p = parameters(force_start=0.333, force_end=0.334, max_step=2)
    r = simulate(p, Load([0.333, 0.334], [200, 200]))
    assert r.lift_off == pytest.approx(0.333)
    assert r.touchdown == pytest.approx(0.333 + 0.001 * (2 + np.sqrt(2)), abs=1e-10)


def test_short_triangular_pulse_is_not_skipped():
    r = simulate(parameters(max_step=2), Load([0, 0.5000, 0.5001, 0.5002, 1], [0, 0, 200, 0, 0]))
    assert r.lift_off == pytest.approx(0.50005)
    assert r.touchdown is not None
    assert r.touchdown > r.lift_off


def test_no_return_before_time_limit():
    r = simulate(parameters(end_time=1), constant(200))
    assert r.detached and r.touchdown is None
    assert r.stop_time == 1


def test_downward_force_during_initial_flight():
    r = simulate(parameters(initial_height=10), constant(-100))
    assert r.touchdown == pytest.approx(1, abs=1e-10)
    assert r.pieces[-1].solution(1)[1] == pytest.approx(-20)


def test_return_at_load_boundary():
    r = simulate(parameters(initial_height=5), constant(0))
    assert r.touchdown == pytest.approx(1)
    assert r.stop_time == pytest.approx(1)


def test_large_step_cannot_skip_ground_crossing_and_reascent():
    p = parameters(initial_height=1, initial_velocity=-2, force_end=3, end_time=3, max_step=3)
    r = simulate(p, Load([0, 3], [119.99, 119.99]))
    acceleration = 119.99 / 10 - 10
    expected = (2 - np.sqrt(4 - 2 * acceleration)) / acceleration
    assert r.touchdown == pytest.approx(expected, abs=1e-10)
    assert r.pieces[-1].solution(r.touchdown)[1] < 0


def test_step_refinement_against_analytic_solution():
    # Polynomial RHS is integrated to rounding accuracy by RK45: compare to the
    # reference at every refinement, not a misleading strict monotonic error test.
    errors = []
    for step in (0.4, 0.2, 0.1, 0.05):
        r = simulate(parameters(max_step=step), constant(200))
        errors.append(abs(r.touchdown - (2 + np.sqrt(2))))
    assert max(errors) < 1e-10


@pytest.mark.parametrize("changes", [
    {"mass": 0}, {"gravity": -1}, {"initial_height": -1},
    {"initial_velocity": -1}, {"force_start": -1}, {"force_end": 6},
    {"force_end": 0}, {"end_time": float("inf")}, {"max_step": 0},
    {"output_step": 0}, {"rtol": 0}, {"atol_z": -1}, {"mass": True},
])
def test_bad_parameters(changes):
    with pytest.raises(InputError):
        parameters(**changes)


@pytest.mark.parametrize("t,f", [([0], [0]), ([0, 0], [1, 2]), ([1, 0], [1, 2]),
                                      ([0, 1], [0, np.nan]), ([-1, 1], [0, 0])])
def test_invalid_loads(t, f):
    with pytest.raises(InputError):
        Load(t, f)


def test_missing_coverage():
    with pytest.raises(InputError, match="покрывать"):
        simulate(parameters(), Load([0.1, 1], [0, 0]))


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "config.json"


def test_config_si_conversion(tmp_path):
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    data["parameters"]["m"] = {"value": 10000, "unit": "g"}
    data["parameters"]["t1"] = {"value": 700, "unit": "ms"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_config(path) == load_config(EXAMPLE)
    simulate(load_config(path), load_force(EXAMPLE.with_name("force.csv")))


@pytest.mark.parametrize("mutation", ["missing", "wrong_unit", "nan", "unknown", "no_unit"])
def test_config_rejects_bad_data(tmp_path, mutation):
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    if mutation == "missing":
        del data["parameters"]["m"]
    elif mutation == "wrong_unit":
        data["parameters"]["m"]["unit"] = "N"
    elif mutation == "nan":
        data["parameters"]["m"]["value"] = float("nan")
    elif mutation == "unknown":
        data["parameters"]["engine_thrust"] = 100
    else:
        del data["parameters"]["m"]["unit"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(InputError):
        load_config(path)


def test_csv_units_and_dimension_errors(tmp_path):
    path = tmp_path / "load.csv"
    path.write_text("time[ms],Fz[kN]\n0,0.2\n1000,0.2\n", encoding="utf-8")
    load = load_force(path)
    np.testing.assert_allclose(load.time, [0, 1])
    np.testing.assert_allclose(load.force, [200, 200])
    for bad in ("time[s],Fz[kg]\n0,1\n1,1", "time[s],Fz[N]\n0,abc\n1,1", "time[s],Fz[N]\n0,\n1,1"):
        path.write_text(bad, encoding="utf-8")
        with pytest.raises(InputError):
            load_force(path)
