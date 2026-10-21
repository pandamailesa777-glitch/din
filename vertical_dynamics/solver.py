"""Hybrid contact/free-flight solver, split at every force discontinuity/knot."""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .model import free_acceleration, lift_time


@dataclass
class FlightPiece:
    start: float
    end: float
    solution: object
    acceleration_start: float
    acceleration_slope: float

    def acceleration(self, t):
        return self.acceleration_start + self.acceleration_slope * (t - self.start)


@dataclass
class Simulation:
    parameters: object
    load: object
    pieces: list
    lift_off: float | None
    touchdown: float | None
    initially_airborne: bool
    stop_time: float
    critical_times: np.ndarray

    @property
    def detached(self):
        return self.initially_airborne or self.lift_off is not None


def simulate(p, load):
    load.validate_coverage(p)
    knots = np.unique(np.r_[0.0, p.force_start, p.force_end, p.end_time,
                            load.time[(load.time > p.force_start) & (load.time < p.force_end)]])
    y = np.array([p.initial_height, p.initial_velocity], dtype=float)
    airborne = p.initial_height > 0 or p.initial_velocity > 0
    initial = p.initial_height > 0
    lift = 0.0 if p.initial_height == 0 and p.initial_velocity > 0 else None
    flight_start = 0.0 if airborne else None
    touchdown = None
    pieces, critical = [], [0.0]
    for left, right in zip(knots[:-1], knots[1:]):
        force, slope = load.linear_piece(left, right, p)
        a, jerk = free_acceleration(force, p.mass, p.gravity), slope / p.mass
        if not airborne:
            start = lift_time(left, right, a, jerk)
            if start is None:
                critical.append(right)
                continue
            lift = flight_start = start
            airborne = True
        else:
            start = left
        a_start = a + jerk * (start - left)
        # A solver event can miss two crossings inside one step. Split at every
        # velocity zero so each integration interval has monotonic height.
        cuts = [start, right]
        polynomial = np.trim_zeros([0.5 * jerk, a_start, float(y[1])], "f")
        if len(polynomial) > 1:
            for root in np.roots(polynomial):
                if abs(root.imag) < 1e-10 and 0 < root.real < right - start:
                    cuts.append(start + float(root.real))
        cuts = np.unique(cuts)
        for sub_start, sub_end in zip(cuts[:-1], cuts[1:]):
            sub_a = a + jerk * (sub_start - left)

            def rhs(t, state):
                return [state[1], sub_a + jerk * (t - sub_start)]

            def ground(t, state):
                # Exclude the initial z=0 root at lift-off.
                if t == flight_start and state[0] == 0:
                    return np.finfo(float).tiny
                return state[0]

            ground.terminal = True
            ground.direction = -1
            sol = solve_ivp(rhs, (sub_start, sub_end), y, method="RK45", events=ground,
                            dense_output=True, max_step=p.max_step, rtol=p.rtol,
                            atol=[p.atol_z, p.atol_v])
            if not sol.success:
                raise RuntimeError(f"Интегрирование не завершено: {sol.message}")
            end = float(sol.t[-1])
            piece = FlightPiece(sub_start, end, sol.sol, sub_a, jerk)
            pieces.append(piece)
            critical.extend([sub_start, end])
            if jerk != 0:
                root = -sub_a / jerk
                if 0 < root < end - sub_start:
                    critical.append(sub_start + root)
            y = sol.y[:, -1]
            if len(sol.t_events[0]):
                touchdown = end
                break
        if touchdown is not None:
            break
    stop = touchdown if touchdown is not None else p.end_time
    critical.append(stop)
    return Simulation(p, load, pieces, lift, touchdown, initial, stop, np.unique(critical))
