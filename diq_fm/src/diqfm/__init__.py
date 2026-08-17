"""Drift-Innovation Quotient Flow Matching research implementation."""

from .synthetic import ShearNet, make_params, sample, train_shear

__all__ = ["ShearNet", "make_params", "sample", "train_shear"]
