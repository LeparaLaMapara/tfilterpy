"""
TFilterPy: A Python package for state estimation, including Kalman Filters,
Particle Filters, and Nonlinear Filters.
"""

from .linear_filters import  DaskKalmanFilter
from .nonlinear_filters import DaskNonLinearKalmanFilter
from .particle_filters import DaskParticleFilter

__version__ = "0.0.1"
