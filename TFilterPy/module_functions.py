import numpy as np

class KalmanFilter:
    r"""
    KalmanFilter class for filtering time series data using the Kalman Filter algorithm.

    Example Usage:
    --------------
        import numpy as np
        from KalmanFilter import KalmanFilter

        # Define the multivariate time series data
        data = np.array([[1, 1, 1],
                        [2, 2, 2],
                        [3, 3, 3],
                        [4, 4, 4],
                        [5, 5, 5],
                        [6, 6, 6],
                        [7, 7, 7],
                        [8, 8, 8],
                        [9, 9, 9],
                        [10, 10, 10]])

        # Define the Kalman filter parameters
        F = np.array([[1, 0.1, 0.01],
                    [0, 1, 0.1],
                    [0, 0, 1]])
        H = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
        Q = np.eye(3) * 0.01
        R = np.eye(3) * 0.1
        x0 = np.array([0, 0, 0])
        P0 = np.eye(3) * 1

        # Initialize the Kalman filter
        kf = KalmanFilter(F, H, Q, R, x0, P0)

        # Run the Kalman filter on the data
        state_estimates = kf.run(data)


    Parameters:
    -----------
        F : numpy.ndarray
        State transition matrix.
        H : numpy.ndarray
        Observation matrix.
        Q : numpy.ndarray
        Process noise covariance matrix.
        R : numpy.ndarray
        Observation noise covariance matrix.
        x0 : numpy.ndarray
        Initial state estimate.
        P0 : numpy.ndarray
        Initial state covariance estimate.

    Returns:
    --------
        KalmanFilter object
        Returns an instance of the KalmanFilter class.

    Methods:
    --------
        predict()
        Predict the next state estimate and covariance matrix.
        update(z)
        Update the state estimate and covariance matrix using the observation z.
        run(measurements)
        Run the Kalman Filter algorithm on a set of measurements.

    Raises:
    -------
        ValueError
        Raised when the dimensions of the input matrices are not valid.
    """
    def __init__(self, F: np.ndarray, H: np.ndarray, Q: np.ndarray, R: np.ndarray, x0: np.ndarray, P0: np.ndarray):

        # Type checking
        assert isinstance(F, np.ndarray), "F should be a numpy ndarray"
        assert isinstance(H, np.ndarray), "H should be a numpy ndarray"
        assert isinstance(Q, np.ndarray), "Q should be a numpy ndarray"
        assert isinstance(R, np.ndarray), "R should be a numpy ndarray"
        assert isinstance(x0, np.ndarray), "x0 should be a numpy ndarray"
        assert isinstance(P0, np.ndarray), "P0 should be a numpy ndarray"
        
        # Dimension checking
        assert F.ndim == 2, "F should have 2 dimensions"
        assert H.ndim == 2, "H should have 2 dimensions"
        assert Q.ndim == 2, "Q should have 2 dimensions"
        assert R.ndim == 2, "R should have 2 dimensions"
        assert x0.ndim == 1, "x0 should have 1 dimension"
        assert P0.ndim == 2, "P0 should have 2 dimensions"

        self.F = F   # state transition matrix
        self.H = H   # observation matrix
        self.Q = Q   # process noise covariance matrix
        self.R = R   # observation noise covariance matrix
        self.x = x0  # initial state estimate
        self.P = P0  # initial state covariance estimate
        
        
    def predict(self):
        """
        Performs the prediction step of the Kalman filter.
        """
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
    def update(self, z: np.ndarray)-> None:
        """
        Performs the update step of the Kalman filter.

        Args:
            z (np.ndarray): observation
        """
        # Type checking
        assert isinstance(z, np.ndarray), "z should be a numpy ndarray"
        assert z.ndim == 1, "z should have 1 dimension"

        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = np.dot((np.eye(self.P.shape[0]) - np.dot(K, self.H)), self.P)
        
    def run(self, measurements: np.ndarray)-> np.ndarray:
        """ 
        Args:
            measurements (np.ndarray): array of observations

        Returns:
            np.ndarray: array of state estimates
        """
        # Type checking
        assert isinstance(measurements, np.ndarray), "measurements should be a numpy ndarray"
        assert measurements.ndim == 2, "measurements should have 2 dimensions"

        self.x = np.array(self.x)
        self.P = np.array(self.P)
        
        state_estimates = []
        for z in measurements:
            self.predict()
            self.update(z)
            state_estimates.append(self.x)
        return np.array(state_estimates)
    

class ParticleFilter:
    # TODO
    pass