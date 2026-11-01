"""Grade feasible designs by their measured cycles relative to the exact optimum."""

from sim2gym.packs import accel


def grade_design(task: dict, design) -> dict:
    error = accel.validate_design(design, task["pe_budget"])
    if error:
        return dict(score=0.0, solved=False, feasible=False, cycles=None, error=error)
    result = accel.simulate(task["layer"], **design)
    score = min(1.0, max(0.0, task["optimum"]["value"] / result["cycles"]))
    return dict(score=score, solved=score >= 0.95, feasible=True, cycles=result["cycles"], error=None)
