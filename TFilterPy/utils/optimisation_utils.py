import numpy as np
import dask.array as da

from TFilterPy.base_estimator import BaseEstimator


class ParameterEstimator(BaseEstimator):
    r"""
    A class for estimating Kalman Filter parameters, such as process noise covariance (Q)
    and observation noise covariance (R), using different estimation methods.

    Parameter estimation is crucial for optimal performance of Kalman Filters, particularly
    in dynamic systems where noise characteristics are unknown or change over time.

    Estimation strategies include:
    - Residual Analysis
    - Maximum Likelihood Estimation (MLE)
    - Cross-Validation
    - Adaptive Filtering

    References:
    - Welch, G., & Bishop, G. (1995). An Introduction to the Kalman Filter.
    - Haykin, S. (2001). Kalman Filtering and Neural Networks.
    """

    def __init__(self, estimation_strategy: str = "residual_analysis"):
        r"""
        Initialize the ParameterEstimator with the desired estimation strategy.

        Parameters
        ----------
        estimation_strategy : str, optional
            The strategy to use for estimating parameters. Options include:
            - "residual_analysis": Estimate based on residuals after running the filter.
            - "mle": Maximum Likelihood Estimation (iterative approach).
            - "cross_validation": Perform k-fold cross-validation for best Q and R.
            - "adaptive_filtering": Dynamically update Q and R based on measurement residuals.

        Raises
        ------
        ValueError
            If an invalid estimation strategy is specified.
        """
        super().__init__(name="KalmanFilter")
        valid_strategies = {
            "residual_analysis",
            "mle",
            "cross_validation",
            "adaptive_filtering",
        }
        if estimation_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid strategy: {estimation_strategy}. Must be one of {valid_strategies}."
            )
        self.estimation_strategy = estimation_strategy

    def estimate_with_residual_analysis(self, measurements: da.Array) -> tuple:
        r"""
        Estimate process (Q) and observation (R) noise covariances using residual analysis.

        Parameters
        ----------
        measurements : da.Array, shape (n_timesteps, n_features)
            Observed measurements over time.

        Returns
        -------
        Q : da.Array, shape (n_features, n_features)
            Estimated process noise covariance matrix.
        R : da.Array, shape (n_features, n_features)
            Estimated observation noise covariance matrix.

        Notes
        -----
        - Residual analysis involves analyzing the discrepancies between measurements and predictions.
        - Q is derived from the covariance of the residuals, while R is scaled based on residual variance.
        """
        state_estimates, residuals = self.run_filter(
            measurements
        )  # Call `run_filter` from the child class
        Q = da.cov(residuals, rowvar=False)
        R = da.eye(self.R.shape[0], chunks=self.R.chunks[0][0]) * da.var(residuals)
        return Q, R

    def estimate_with_mle(
        self, measurements: da.Array, max_iterations: int = 5
    ) -> tuple:
        r"""
        Estimate Q and R using Maximum Likelihood Estimation (MLE).

        Parameters
        ----------
        measurements : da.Array, shape (n_timesteps, n_features)
            Observed measurements over time.
        max_iterations : int, optional
            Number of iterations to refine the estimates. Default is 5.

        Returns
        -------
        Q : da.Array, shape (n_features, n_features)
            Estimated process noise covariance matrix.
        R : da.Array, shape (n_features, n_features)
            Estimated observation noise covariance matrix.

        Notes
        -----
        - MLE iteratively adjusts Q and R by maximizing the likelihood of the observed measurements.
        - The process is repeated for `max_iterations` to converge towards optimal estimates.
        """
        Q, R = self.Q, self.R

        for _ in range(max_iterations):
            state_estimates, residuals = self.run_filter(measurements)
            Q_new = da.eye(residuals.shape[1], chunks=residuals.chunks[1][0]) * da.var(
                residuals, axis=0
            )
            R_new = da.eye(self.R.shape[0], chunks=self.R.chunks[0][0]) * da.var(
                residuals
            )
            Q, R = Q_new.persist(), R_new.persist()

        return Q, R

    def estimate_with_cross_validation(
        self, measurements: da.Array, k_folds: int = 5
    ) -> tuple:
        r"""
        Estimate Q and R using k-fold cross-validation.

        Parameters
        ----------
        measurements : da.Array, shape (n_timesteps, n_features)
            Observed measurements over time.
        k_folds : int, optional
            Number of folds to split the data for cross-validation. Default is 5.

        Returns
        -------
        best_Q : da.Array, shape (n_features, n_features)
            Best estimated process noise covariance matrix.
        best_R : da.Array, shape (n_features, n_features)
            Best estimated observation noise covariance matrix.

        Notes
        -----
        - The data is split into k-folds. The model is trained on k-1 folds and validated on the remaining fold.
        - The fold with the lowest validation error determines the best Q and R estimates.
        """
        n_samples = measurements.shape[0]
        fold_size = n_samples // k_folds

        fold_scores, fold_Qs, fold_Rs = [], [], []

        for fold in range(k_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size
            train_data = da.concatenate(
                [measurements[:val_start], measurements[val_end:]]
            )
            val_data = measurements[val_start:val_end]

            state_estimates, residuals = self.run_filter(train_data)
            val_score = da.mean((val_data - state_estimates[:fold_size]) ** 2)
            fold_scores.append(val_score)

            Q = da.cov(residuals.T)
            R = da.eye(self.R.shape[0], chunks=self.R.chunks[0][0]) * da.var(residuals)
            fold_Qs.append(Q)
            fold_Rs.append(R)

        fold_scores, fold_Qs, fold_Rs = da.compute(fold_scores, fold_Qs, fold_Rs)
        best_fold = np.argmin(fold_scores)
        return fold_Qs[best_fold], fold_Rs[best_fold]

    def estimate_with_adaptive_filtering(self, measurements: da.Array) -> tuple:
        r"""
        Estimate Q and R adaptively based on innovations over time.

        Parameters
        ----------
        measurements : da.Array, shape (n_timesteps, n_features)
            Observed measurements over time.

        Returns
        -------
        Q : da.Array, shape (n_features, n_features)
            Estimated process noise covariance matrix.
        R : da.Array, shape (n_features, n_features)
            Estimated observation noise covariance matrix.

        Notes
        -----
        - Innovations (differences between predictions and observations) are used to update Q and R adaptively.
        - A smoothing factor (alpha) determines the influence of recent measurements.
        """
        Q, R = self.Q, self.R
        alpha = 0.01
        innovations = measurements - da.dot(self.H, self.x)
        outer_products = da.einsum("ij,ik->ijk", innovations, innovations)
        Q_new = da.mean(outer_products, axis=0).rechunk("auto")
        R_new = da.var(innovations, axis=0).rechunk("auto")

        Q = (1 - alpha) * Q + alpha * Q_new
        R = (1 - alpha) * R + alpha * R_new
        return Q, R

    def estimate_parameters(self, measurements: da.Array):
        r"""
        Estimate the parameters Q and R using the specified estimation strategy.
        """
        return getattr(self, f"estimate_with_{self.estimation_strategy}")(measurements)
