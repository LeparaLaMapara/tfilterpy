import numpy as np
import dask.array as da
import dask

from typing import Union

from TFilterPy.utils.optimisation_utils import ParameterEstimator


class DaskKalmanFilter(ParameterEstimator):
    r"""
    Dask-based implementation of a Kalman Filter that supports distributed computing for
    large datasets. This class extends the ParameterEstimator to estimate the process
    noise covariance (Q) and observation noise covariance (R) while applying Kalman Filtering
    on incoming measurements.

    The Kalman Filter is a recursive algorithm that estimates the state of a linear dynamic
    system from noisy measurements. This implementation leverages Dask to scale computations
    across distributed systems.

    Parameters
    ----------
    state_transition_matrix : np.ndarray or da.Array, shape (n_features, n_features)
        The state transition matrix (F) representing how the system evolves between states.

    observation_matrix : np.ndarray or da.Array, shape (n_observations, n_features)
        The observation matrix (H) that maps the true state space into the observed space.

    process_noise_cov : np.ndarray or da.Array, shape (n_features, n_features)
        Covariance matrix (Q) representing the process noise.

    observation_noise_cov : np.ndarray or da.Array, shape (n_observations, n_observations)
        Covariance matrix (R) representing the observation noise.

    initial_state : np.ndarray or da.Array, shape (n_features,)
        Initial state vector (x0) of the system.

    initial_covariance : np.ndarray or da.Array, shape (n_features, n_features)
        Initial state covariance matrix (P0), representing initial uncertainty in the state.

    estimation_strategy : str, optional, default="residual_analysis"
        The strategy for estimating Q and R. Can be one of:
        - "residual_analysis"
        - "mle"
        - "cross_validation"
        - "adaptive_filtering"

    Raises
    ------
    ValueError
        If matrix dimensions do not conform to Kalman Filter requirements.

    References
    ----------
    Welch, G., & Bishop, G. (1995). An Introduction to the Kalman Filter.
    """

    def __init__(
        self,
        state_transition_matrix: Union[np.ndarray, da.Array],
        observation_matrix: Union[np.ndarray, da.Array],
        process_noise_cov: Union[np.ndarray, da.Array],
        observation_noise_cov: Union[np.ndarray, da.Array],
        initial_state: Union[np.ndarray, da.Array],
        initial_covariance: Union[np.ndarray, da.Array],
        estimation_strategy: str = "residual_analysis",
    ):

        # Initialize the parent class (ParameterEstimator)
        super().__init__(estimation_strategy)

        # Input validation and conversion to Dask arrays
        if state_transition_matrix.shape[0] != state_transition_matrix.shape[1]:
            raise ValueError("State transition matrix (F) must be square.")
        if observation_matrix.shape[1] != state_transition_matrix.shape[0]:
            raise ValueError(
                "Observation matrix (H) dimensions must be compatible with F."
            )
        if process_noise_cov.shape != state_transition_matrix.shape:
            raise ValueError("Process noise covariance (Q) dimensions must match F.")
        if observation_noise_cov.shape[0] != observation_noise_cov.shape[1]:
            raise ValueError("Observation noise covariance (R) must be square.")
        if initial_state.shape[0] != state_transition_matrix.shape[0]:
            raise ValueError("Initial state (x0) dimensions must match F.")
        if initial_covariance.shape != state_transition_matrix.shape:
            raise ValueError("Initial covariance (P0) dimensions must match F.")

        # Convert to Dask arrays if necessary
        self.F = (
            da.from_array(state_transition_matrix, chunks=state_transition_matrix.shape)
            if isinstance(state_transition_matrix, np.ndarray)
            else state_transition_matrix
        )
        self.H = (
            da.from_array(observation_matrix, chunks=observation_matrix.shape)
            if isinstance(observation_matrix, np.ndarray)
            else observation_matrix
        )
        self.Q = (
            da.from_array(process_noise_cov, chunks=process_noise_cov.shape)
            if isinstance(process_noise_cov, np.ndarray)
            else process_noise_cov
        )
        self.R = (
            da.from_array(observation_noise_cov, chunks=observation_noise_cov.shape)
            if isinstance(observation_noise_cov, np.ndarray)
            else observation_noise_cov
        )
        self.x = (
            da.from_array(initial_state, chunks=initial_state.shape)
            if isinstance(initial_state, np.ndarray)
            else initial_state
        )
        self.P = (
            da.from_array(initial_covariance, chunks=initial_covariance.shape)
            if isinstance(initial_covariance, np.ndarray)
            else initial_covariance
        )

    def fit(self, X: Union[np.ndarray, da.Array]) -> "DaskKalmanFilter":
        r"""
        Prepare the Kalman Filter by storing the measurements as a Dask array.

        Parameters
        ----------
        X : np.ndarray or da.Array, shape (n_timesteps, n_observations)
            Array of measurements over time.

        Returns
        -------
        self : DaskKalmanFilter
            The fitted Kalman Filter instance.

        Raises
        ------
        ValueError
            If the input measurements are not 2-dimensional.
        """
        if X.ndim != 2:
            raise ValueError(
                f"Measurements must be a 2D array. Received array with shape {X.shape}."
            )

        if isinstance(X, da.Array):
            self.X = X
        else:
            self.X = da.from_array(X, chunks=(X.shape[0] // 4, X.shape[1]))
        return self

    def predict(self) -> da.Array:
        r"""
        Perform state estimation over all time steps using the Kalman Filter algorithm.

        This method constructs a Dask computation graph to process the entire measurement
        sequence in parallel using delayed execution.

        Returns
        -------
        state_estimates : da.Array, shape (n_timesteps, n_features)
            The estimated state at each time step.

        Notes
        -----
        - The Kalman Filter operates in two steps: prediction and update.
        - Predictions are made using the state transition matrix F.
        - Updates are performed using the observation matrix H and Kalman Gain K.
        - This method leverages Dask to parallelize the filter process over multiple time steps.
        """

        @dask.delayed
        def kalman_step(i, x, P, F, H, Q, R, measurements):
            x = da.dot(F, x)
            P = da.dot(da.dot(F, P), F.T) + Q

            y = measurements[i] - da.dot(H, x)
            S = da.dot(da.dot(H, P), H.T) + R
            K = da.dot(da.dot(P, H.T), da.linalg.inv(S))
            x = x + da.dot(K, y)
            I = da.eye(P.shape[0], chunks=P.chunks[0][0])
            P = da.dot(I - da.dot(K, H), P)
            return (x, P)

        n_timesteps = self.X.shape[0]
        state_estimates = []
        x, P = self.x, self.P

        for i in range(n_timesteps):
            delayed_result = kalman_step(
                i, x, P, self.F, self.H, self.Q, self.R, self.X
            )
            x, P = dask.compute(delayed_result)[0]
            state_estimates.append(x)

        state_estimates = da.stack(state_estimates, axis=0)
        return state_estimates

    def run_filter(self, measurements: da.Array) -> tuple:
        r"""
        Apply the Kalman Filter on measurements to compute state estimates and residuals.

        Parameters
        ----------
        measurements : da.Array, shape (n_timesteps, n_observations)
            Observed measurements over time.

        Returns
        -------
        state_estimates : da.Array, shape (n_timesteps, n_features)
            Estimated states over the measurement timeline.
        residuals : da.Array, shape (n_timesteps, n_observations)
            Difference between observed and predicted measurements (innovations).

        Notes
        -----
        - This function is used by parameter estimation strategies to compute residuals.
        - Residuals are used for adaptive filtering and cross-validation strategies.
        """
        n_timesteps = measurements.shape[0]
        state_estimates = []
        residuals = []

        x, P = self.x, self.P

        for i in range(n_timesteps):
            x = da.dot(self.F, x)
            P = da.dot(da.dot(self.F, P), self.F.T) + self.Q

            y = measurements[i] - da.dot(self.H, x)
            S = da.dot(da.dot(self.H, P), self.H.T) + self.R
            K = da.dot(da.dot(P, self.H.T), da.linalg.inv(S))
            x = x + da.dot(K, y)
            I = da.eye(P.shape[0], chunks=P.chunks[0][0])
            P = da.dot(I - da.dot(K, self.H), P)

            state_estimates.append(x)
            residuals.append(y)

        state_estimates = da.stack(state_estimates, axis=0)
        residuals = da.stack(residuals, axis=0)
        return state_estimates, residuals

    def estimate_parameters(self, measurements: da.Array) -> tuple:
        """
        Estimate process (Q) and observation (R) noise covariances using the specified strategy.

        Parameters
        ----------
        measurements : da.Array, shape (n_timesteps, n_observations)
            Observed measurements over time.

        Returns
        -------
        Q : da.Array, shape (n_features, n_features)
            Estimated process noise covariance matrix.
        R : da.Array, shape (n_features, n_features)
            Estimated observation noise covariance matrix.

        Notes
        -----
        - This method calls the appropriate estimation strategy from the parent class.
        - The available strategies include residual analysis, MLE, cross-validation,
          and adaptive filtering.
        """
        return super().estimate_parameters(measurements)
