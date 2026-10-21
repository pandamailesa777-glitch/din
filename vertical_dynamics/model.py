"""Contact law and the free-flight equations, independent of input/output."""


def free_acceleration(force, mass, gravity):
    return force / mass - gravity


def contact_reaction(force, mass, gravity):
    return max(mass * gravity - force, 0.0)


def lift_time(left, right, acceleration_left, acceleration_slope):
    """First boundary of a strictly positive acceleration interval, or None."""
    if acceleration_left > 0:
        return left
    if acceleration_slope > 0:
        candidate = left - acceleration_left / acceleration_slope
        if left <= candidate < right:
            return candidate
    return None
