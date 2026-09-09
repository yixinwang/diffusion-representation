"""Frozen feature-space diagnostics; no feature learning or data loading."""
import numpy as np


def _features(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) < 2 or not np.isfinite(value).all():
        raise ValueError('finite matrix with at least two observations/features required')
    return value


def polynomial_kid(real, fake):
    """Unbiased full-bank MMD^2, (x.y/d + 1)^3; unequal n,m supported."""
    real, fake = _features(real), _features(fake)
    if real.shape[1] != fake.shape[1]:
        raise ValueError('feature dimensions differ')
    n, m, d = len(real), len(fake), real.shape[1]
    rr = (real @ real.T / d + 1) ** 3
    ff = (fake @ fake.T / d + 1) ** 3
    rf = (real @ fake.T / d + 1) ** 3
    result = ((rr.sum()-np.trace(rr))/(n*(n-1))
              + (ff.sum()-np.trace(ff))/(m*(m-1)) - 2*rf.mean())
    if not np.isfinite(result):
        raise FloatingPointError('nonfinite KID')
    return float(result)


def squared_distances(a, b):
    value = (np.sum(a*a, axis=1)[:, None] + np.sum(b*b, axis=1)[None, :]
             - 2*a @ b.T)
    # Exact squared distances are nonnegative; remove cancellation roundoff.
    value = np.maximum(value, 0)
    # Equal nonzero rows can otherwise acquire positive roundoff distances,
    # creating spurious nonzero duplicate radii under the strict rule.
    _, identity = np.unique(np.concatenate((a,b)),axis=0,return_inverse=True)
    value[identity[:len(a),None] == identity[len(a):][None,:]] = 0
    return value


def prdc(real, fake, nearest_k=5):
    """PRDC Euclidean neighborhoods, strict <, k excludes each point itself."""
    real, fake = _features(real), _features(fake)
    if real.shape[1] != fake.shape[1] or not 1 <= nearest_k < min(len(real), len(fake)):
        raise ValueError('invalid dimensions or k')
    rr, ff = squared_distances(real, real), squared_distances(fake, fake)
    np.fill_diagonal(rr, np.inf)
    np.fill_diagonal(ff, np.inf)
    r = np.partition(rr, nearest_k-1, axis=1)[:, nearest_k-1]
    f = np.partition(ff, nearest_k-1, axis=1)[:, nearest_k-1]
    rf = squared_distances(real, fake)
    inside_r, inside_f = rf < r[:, None], rf < f[None, :]
    return {'precision':float(inside_r.any(axis=0).mean()),
            'recall':float(inside_f.any(axis=1).mean()),
            'density':float(inside_r.sum(axis=0).mean()/nearest_k),
            'coverage':float((rf.min(axis=1) < r).mean()),
            'nearest_k':nearest_k,
            'real_duplicate_rows':len(real)-len(np.unique(real, axis=0)),
            'fake_duplicate_rows':len(fake)-len(np.unique(fake, axis=0)),
            'cross_radius_ties_real':int((rf == r[:, None]).sum()),
            'cross_radius_ties_fake':int((rf == f[None, :]).sum())}


def evaluate(real, fake):
    return {'real_count':len(real),'fake_count':len(fake),
            'kid_unbiased_full_bank':polynomial_kid(real, fake), **prdc(real, fake)}
