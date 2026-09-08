"""Deterministic iteration-5 algebra and resource checks; never loads observations.

Audited repository pin: ffec36e35c325d36bc9cdfc73b04305f6f485205.
Numbers produced here evaluate sufficient bounds, not empirical sample complexity.
No training, simulation, scheduler submission, or benchmark data access is performed.
"""
from __future__ import annotations
import json
import math
from fractions import Fraction
from typing import Iterable, Mapping

PIN = "ffec36e35c325d36bc9cdfc73b04305f6f485205"


def _integer(name: str, value: int, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def h_moments() -> tuple[Fraction, Fraction]:
    """Exact integrals of h and h^2, h(x)=-4x^3+6x^2-1, over [0,1]."""
    coefficients = [Fraction(-1), Fraction(0), Fraction(6), Fraction(-4)]
    mean = sum((a / (i + 1) for i, a in enumerate(coefficients)), Fraction(0))
    square = sum((a*b / (i+j+1) for i, a in enumerate(coefficients)
                  for j, b in enumerate(coefficients)), Fraction(0))
    return mean, square


def dependence_lower(rho: float) -> float:
    """Joint KL floor for one pair versus products on the SAME fixed coordinates.

    Not a lower bound for learned invertible charts or dependent latent decoders.
    """
    rho = _finite("rho", rho)
    if not 0 < abs(rho) < 1:
        raise ValueError("rho must have magnitude strictly between zero and one")
    return rho*rho * float(h_moments()[1])**2 / (2*(1+abs(rho)))


def pro4_bound(n: int, b: int, s: int, m: float, M: float, H: float,
               *, catalogue: int = 1, delta: float = 0.05,
               tau: float = 0.0) -> dict[str, float | int]:
    """User-supplied Pro4 sufficient bound for ONE conditional density table.

    P counts all coefficients, not independent pixels. Group/model counts must
    be handled separately. It does not include context discovery, selection,
    omitted-context bias, chart error, or a learned-neural optimization bound.
    """
    _integer("n", n); _integer("b", b, 2); _integer("s", s, 0)
    _integer("catalogue", catalogue)
    m, M, H, delta, tau = map(float, (m, M, H, delta, tau))
    if not all(map(math.isfinite, (m, M, H, delta, tau))):
        raise ValueError("all scalar inputs must be finite")
    if not (0 < m <= 1 <= M and H >= 0 and 0 < delta < 1 and tau >= 0):
        raise ValueError("invalid density, Hessian, confidence, or gap bounds")
    if b*b < H/6:
        raise ValueError("approximation condition b^2 >= H/6 fails")
    ell, upper = m/2, 2*M
    P = (b+2)**s*(b+1)
    A = H*(9*s+1)/2 + M*H/6
    kappa = 16*(upper/ell)**2 + 8*math.log(upper/ell)/3
    approximation = A*A/(ell*b**4)
    entropy = P*math.log1p(8*(upper-ell)*n/ell)+math.log(catalogue/delta)
    estimation = kappa*entropy/n + 2*tau + 1/n
    return dict(n=n, b=b, s=s, coefficients=P, ell=ell, upper=upper,
                approximation_constant=A, kappa=kappa,
                approximation=approximation, estimation=estimation,
                total=approximation+estimation)


def table_resources(s: int, b: int = 8, groups: int = 1,
                    bytes_per_scalar: int = 4) -> dict[str, int | float]:
    """Raw coefficient memory only; no CDF cache, optimizer or activations."""
    _integer("s", s, 0); _integer("b", b, 2)
    _integer("groups", groups); _integer("bytes_per_scalar", bytes_per_scalar)
    per_group = (b+2)**s*(b+1)
    count = groups*per_group
    return dict(coefficients_per_group=per_group, coefficients=count,
                raw_bytes=count*bytes_per_scalar,
                raw_gib=count*bytes_per_scalar/(2**30),
                active_density_products_upper=2*3**s)


def invert_linear_density(a: float, right: float, width: float,
                          mass_from_left: float) -> float:
    """Invert v=a*t+0.5*d*t^2 with d=(right-a)/width, 0<=t<=width.

    Returns distance from the left knot, NOT a unit-bin fraction. No clipping
    of invalid probabilities or negative discriminants is permitted.
    """
    a, right, width, v = map(float, (a, right, width, mass_from_left))
    if not all(map(math.isfinite, (a, right, width, v))):
        raise ValueError("nonfinite inverse input")
    if min(a, right, width) <= 0:
        raise ValueError("endpoint densities and interval width must be positive")
    interval_mass = width*(a+right)/2
    if not 0 <= v <= interval_mass:
        raise ValueError("target probability is outside this interval")
    slope = (right-a)/width
    discriminant = a*a + 2*slope*v
    if discriminant <= 0:
        raise ArithmeticError("positive endpoint densities require positive discriminant")
    return 2*v/(a+math.sqrt(discriminant))


def validation_selection_penalty(lower: float, upper: float, candidates: int,
                                 n_validation: int, delta: float = 0.05) -> float:
    """Two-sided Hoeffding oracle penalty for INDEPENDENT validation arrays.

    All candidate fitting and proposal choices must be independent of validation.
    Not valid for adaptively reused repair data as fresh confirmation.
    """
    _integer("candidates", candidates); _integer("n_validation", n_validation)
    if not (0 < lower <= upper and 0 < delta < 1):
        raise ValueError("invalid range or delta")
    return 2*math.log(upper/lower)*math.sqrt(
        math.log(2*candidates/delta)/(2*n_validation))


def verify_split_roles(roles: Mapping[str, Iterable[str]],
                       expected: Mapping[str, int]) -> None:
    """Validate supplied EXISTING manifests, never invent or reshuffle IDs."""
    if set(roles) != set(expected):
        raise ValueError("split-role names differ from the declared existing manifest")
    seen: set[str] = set()
    for role, expected_count in expected.items():
        _integer(f"count for {role}", expected_count, 0)
        ids = list(roles[role])
        if len(ids) != expected_count or len(set(ids)) != len(ids):
            raise ValueError(f"wrong count or duplicate ID in {role}")
        if seen.intersection(ids):
            raise ValueError("an independent source group occurs in multiple roles")
        seen.update(ids)


def validate_run_contract(record: Mapping[str, object]) -> None:
    """Reject common attribution/data-access errors before a development run.

    Flags are provenance assertions to be independently checked, not proof that
    a runner obeyed them. This function does not access any data or scheduler.
    """
    required_true = ("synthetic_correctness_passed", "global_dependent_latent_control",
                     "all_prior_coordinates_counted", "all_fit_costs_charged",
                     "existing_split_manifests_verified", "copied_control_bitwise_equal")
    required_false = ("official_tests_accessed", "excluded_discovery_accessed",
                      "reused_development_called_fresh", "pixel_independence_assumed",
                      "table_bound_claimed_for_unmodified_neural_pilot")
    for name in required_true:
        if record.get(name) is not True:
            raise ValueError(f"required verified flag missing/false: {name}")
    for name in required_false:
        if record.get(name) is not False:
            raise ValueError(f"forbidden or unspecified condition: {name}")
    n = record.get("independent_fit_arrays")
    _integer("independent_fit_arrays", n)  # type: ignore[arg-type]
    if record.get("statistical_unit") != "complete_array_or_original_source_group":
        raise ValueError("invalid statistical unit")


def report() -> dict:
    rho, H = 0.5, 10.5  # H=21*rho is a conservative valid Hessian bound.
    rows = []
    for n in (300, 20000, 40000, 10**9, 10**10):
        options = [pro4_bound(n,b,1,0.5,1.5,H) for b in range(2,512)]
        rows.append(min(options, key=lambda row: row["total"]))
    return {
        "kind": "deterministic_bound_evaluation_not_an_experiment", "reviewed_pin": PIN,
        "h_mean": str(h_moments()[0]), "h_second_moment": str(h_moments()[1]),
        "rho": rho, "one_pair_joint_gap_lower": dependence_lower(rho),
        "b2_n40000": pro4_bound(40000,2,1,0.5,1.5,H),
        "integer_search_b_2_to_511": rows,
        "raw_float32_table_memory": {str(s): table_resources(s) for s in (1,2,4,8)},
        "cifar_per_coordinate_tables_s8": table_resources(8, groups=3072),
        "warning": "A vacuous sufficient bound is not a lower bound on learnability."
    }


if __name__ == "__main__":
    print(json.dumps(report(), indent=2, allow_nan=False))
