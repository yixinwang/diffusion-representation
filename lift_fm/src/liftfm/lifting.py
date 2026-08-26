from __future__ import annotations

import math
import numpy as np


def haar2d_forward(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-level orthonormal 2D Haar transform.

    Parameters
    ----------
    images: [N,H,W], H and W even.

    Returns
    -------
    coarse: [N,H/2,W/2]
    details: [N,H/2,W/2,3] ordered (HL, LH, HH).
    """
    x = np.asarray(images, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] % 2 or x.shape[2] % 2:
        raise ValueError("images must be [N,H,W] with even H,W")
    rt2 = math.sqrt(2.0)
    low_w = (x[:, :, 0::2] + x[:, :, 1::2]) / rt2
    high_w = (x[:, :, 0::2] - x[:, :, 1::2]) / rt2
    ll = (low_w[:, 0::2, :] + low_w[:, 1::2, :]) / rt2
    hl = (low_w[:, 0::2, :] - low_w[:, 1::2, :]) / rt2
    lh = (high_w[:, 0::2, :] + high_w[:, 1::2, :]) / rt2
    hh = (high_w[:, 0::2, :] - high_w[:, 1::2, :]) / rt2
    return ll, np.stack((hl, lh, hh), axis=-1)


def haar2d_inverse(coarse: np.ndarray, details: np.ndarray) -> np.ndarray:
    z = np.asarray(coarse, dtype=np.float64)
    r = np.asarray(details, dtype=np.float64)
    if z.ndim != 3 or r.shape != (*z.shape, 3):
        raise ValueError("coarse [N,h,w] and details [N,h,w,3] required")
    hl, lh, hh = (r[..., index] for index in range(3))
    rt2 = math.sqrt(2.0)
    low_even = (z + hl) / rt2
    low_odd = (z - hl) / rt2
    high_even = (lh + hh) / rt2
    high_odd = (lh - hh) / rt2
    low_w = np.empty((len(z), z.shape[1] * 2, z.shape[2]), dtype=np.float64)
    high_w = np.empty_like(low_w)
    low_w[:, 0::2, :] = low_even
    low_w[:, 1::2, :] = low_odd
    high_w[:, 0::2, :] = high_even
    high_w[:, 1::2, :] = high_odd
    output = np.empty((len(z), z.shape[1] * 2, z.shape[2] * 2), dtype=np.float64)
    output[:, :, 0::2] = (low_w + high_w) / rt2
    output[:, :, 1::2] = (low_w - high_w) / rt2
    return output


def pack_coefficients(coarse: np.ndarray, details: np.ndarray) -> np.ndarray:
    z = np.asarray(coarse, dtype=np.float64)
    r = np.asarray(details, dtype=np.float64)
    if z.ndim != 3 or r.shape != (*z.shape, 3):
        raise ValueError("invalid coefficient shapes")
    n, h, w = z.shape
    packed = np.empty((n, 2 * h, 2 * w), dtype=np.float64)
    packed[:, :h, :w] = z
    packed[:, h:, :w] = r[..., 0]
    packed[:, :h, w:] = r[..., 1]
    packed[:, h:, w:] = r[..., 2]
    return packed


def unpack_coefficients(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(packed, dtype=np.float64)
    if y.ndim != 3 or y.shape[1] % 2 or y.shape[2] % 2:
        raise ValueError("packed coefficients must be [N,2h,2w]")
    h, w = y.shape[1] // 2, y.shape[2] // 2
    z = y[:, :h, :w]
    r = np.stack((y[:, h:, :w], y[:, :h, w:], y[:, h:, w:]), axis=-1)
    return z, r
