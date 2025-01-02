"""
TFilterPy: A Python package for state estimation, including Kalman Filters,
Particle Filters, and Nonlinear Filters.
"""

from .state_estimation.linear_filters import DaskKalmanFilter
from .state_estimation.nonlinear_filters import DaskNonLinearKalmanFilter
from .state_estimation.particle_filters import DaskParticleFilter
from .utils.optimisation_utils import ParameterEstimator

__version__ = "0.0.1"
