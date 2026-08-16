"""Static Flow-Induced Quotient Flow Matching (S-FIQ)."""
from .core import (
    OrthogonalChart,
    VectorField,
    BlockGaussianFiber,
    Autoencoder,
    fit_flow_moment_chart,
    refine_fiber_gauge,
    fiber_dependence_scores,
    train_flow,
    sample_flow,
    train_fiber,
    train_autoencoder,
)
