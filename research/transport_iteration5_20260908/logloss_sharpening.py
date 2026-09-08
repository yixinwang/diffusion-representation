"""Sharper convex log-loss constants; deterministic arithmetic, no data access."""
from __future__ import annotations
import json
import math
from audit_checks import pro4_bound, dependence_lower


def logloss_bernstein(B: float) -> float:
    """V_B=B^2/(B-1+exp(-B)), continuous at B=0; positive B only otherwise."""
    B = float(B)
    if not math.isfinite(B) or B < 0:
        raise ValueError("B must be finite and nonnegative")
    if B == 0:
        return 2.0
    if B < 1e-3:
        # The denominator divided by B^2 avoids catastrophic cancellation.
        normalized = 0.5 - B/6 + B*B/24 - B**3/120 + B**4/720 - B**5/5040
        return 1/normalized
    return B*B/(B+math.expm1(-B))


def sharper_bound(n: int, b: int, s: int, m: float, M: float, H: float,
                  *, catalogue: int = 1, delta: float = .05,
                  tau: float = 0.0) -> dict:
    row = pro4_bound(n,b,s,m,M,H,catalogue=catalogue,delta=delta,tau=tau)
    lower, upper, P = row['ell'], row['upper'], row['coefficients']
    B = math.log(upper/lower)
    V = logloss_bernstein(B)
    coefficient = 2*V + 4*B/3
    entropy = P*math.log1p(8*(upper-lower)*n/lower)+math.log(catalogue/delta)
    estimation = coefficient*entropy/n+2*tau+1/n
    return dict(n=n,b=b,s=s,coefficients=P,log_ratio_bound=B,
                bernstein_constant=V,excess_coefficient=coefficient,
                approximation=row['approximation'],estimation=estimation,
                total=row['approximation']+estimation)


def report() -> dict:
    rows = []
    for n in (20000,40000,10**6,10**7,10**8):
        rows.append(min((sharper_bound(n,b,1,.5,1.5,10.5)
                         for b in range(2,1000)),key=lambda r:r['total']))
    return dict(kind="deterministic_sharpened_bound_not_an_experiment",
                pair_joint_gap_lower=dependence_lower(.5),
                search_b_2_to_999=rows,
                warning="Sufficient bound only; no neural-context approximation or empirical superiority is proved.")

if __name__ == '__main__':
    print(json.dumps(report(),indent=2,allow_nan=False))
