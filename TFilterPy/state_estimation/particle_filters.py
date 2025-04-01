import numpy as np
import dask.array as da
from TFilterPy.utils.optimisation_utils import ParameterEstimator

class DaskParticleFilter(ParameterEstimator):
    """
    A multivariate, scalable particle filter using Dask. Inherits parameter estimation 
    methods from ParameterEstimator.
    """
    def __init__(self, state_transition, observation_model, process_noise_cov, 
                 observation_noise_cov, initial_state, num_particles=1000, use_dask=True, 
                 estimation_strategy="residual_analysis"):
        """
        Initialize the DaskParticleFilter. In addition to particle filter parameters, 
        we specify whether to use Dask for scalability and which parameter estimation 
        strategy to use.
        """
        # Initialize the ParameterEstimator
        super().__init__(estimation_strategy=estimation_strategy)
        self.use_dask = use_dask
        self.state_dim = initial_state.shape[0]
        self.num_particles = num_particles

        # Convert inputs to Dask arrays if scalability is desired
        if self.use_dask:
            self.state_transition = da.from_array(state_transition, chunks=state_transition.shape)
            self.observation_model = da.from_array(observation_model, chunks=observation_model.shape)
            self.process_noise_cov = da.from_array(process_noise_cov, chunks=process_noise_cov.shape)
            self.observation_noise_cov = da.from_array(observation_noise_cov, chunks=observation_noise_cov.shape)
            self.initial_state = da.from_array(initial_state, chunks=initial_state.shape)
        else:
            self.state_transition = state_transition
            self.observation_model = observation_model
            self.process_noise_cov = process_noise_cov
            self.observation_noise_cov = observation_noise_cov
            self.initial_state = initial_state

        # Step 1: Initialization - all particles start at the same known state
        particles_np = np.repeat(initial_state.reshape(1, self.state_dim), num_particles, axis=0)
        if self.use_dask:
            self.particles = da.from_array(particles_np, chunks=(num_particles // 10, self.state_dim))
        else:
            self.particles = particles_np
        
        # Uniform weights
        weights_np = np.ones(num_particles) / num_particles
        if self.use_dask:
            self.weights = da.from_array(weights_np, chunks=(num_particles // 10,))
        else:
            self.weights = weights_np

        # Current state estimate (initially set to the initial state)
        self.state = self.initial_state.compute() if self.use_dask else self.initial_state

        # For parameter estimation methods, we store a copy of Q and R (could be updated later)
        # For demonstration, we initialize them as given.
        self.Q = self.process_noise_cov
        self.R = self.observation_noise_cov

    def predict(self):
        """
        Step 2: Prediction. Propagate each particle using the state transition model plus 
        Gaussian process noise.
        """
        noise_np = np.random.multivariate_normal(
            np.zeros(self.state_dim),
            self.process_noise_cov.compute() if self.use_dask else self.process_noise_cov,
            self.num_particles
        )
        if self.use_dask:
            noise = da.from_array(noise_np, chunks=self.particles.chunksize)
        else:
            noise = noise_np
        
        if self.use_dask:
            self.particles = da.dot(self.particles, self.state_transition.T) + noise
            self.particles = self.particles.persist()
        else:
            self.particles = (self.state_transition @ self.particles.T).T + noise

    def update(self, measurement):
        """
        Step 3: Measurement Update. Update particle weights based on the likelihood 
        of the observed measurement.
        
        Parameters:
            measurement (np.ndarray): The observed measurement.
        """
        if self.use_dask and not isinstance(measurement, da.Array):
            measurement = da.from_array(measurement, chunks=measurement.shape)
        
        if self.use_dask:
            predicted_measurements = da.dot(self.particles, self.observation_model.T)
        else:
            predicted_measurements = (self.observation_model @ self.particles.T).T
        
        diff = predicted_measurements - measurement
        
        R_val = self.observation_noise_cov[0, 0].compute() if self.use_dask else self.observation_noise_cov[0, 0]
        if self.use_dask:
            likelihood = da.exp(-0.5 * da.sum(diff**2, axis=1) / R_val)
        else:
            likelihood = np.exp(-0.5 * np.sum(diff**2, axis=1) / R_val)
        
        self.weights = self.weights * likelihood
        self.weights = self.weights + 1e-300  # Avoid zero weights
        self.weights = self.weights / self.weights.sum()
        
        self.resample()
        self.estimate_state()

    def resample(self):
        """
        Step 4: Resampling. Multinomial resampling to refocus on high-probability particles.
        """
        weights_np = self.weights.compute() if self.use_dask else self.weights
        indices = np.random.choice(np.arange(self.num_particles), size=self.num_particles, p=weights_np)
        if self.use_dask:
            particles_np = self.particles.compute()
            particles_resampled = particles_np[indices]
            self.particles = da.from_array(particles_resampled, chunks=self.particles.chunksize)
            self.weights = da.from_array(np.ones(self.num_particles) / self.num_particles, chunks=self.weights.chunksize)
        else:
            self.particles = self.particles[indices]
            self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate_state(self):
        """
        Step 5: State Estimation. Compute the state estimate as the weighted average of particles.
        """
        if self.use_dask:
            self.state = da.average(self.particles, weights=self.weights, axis=0).compute()
        else:
            self.state = np.average(self.particles, weights=self.weights, axis=0)

    def step(self, measurement):
        """
        Step 6: Iteration. Run one full filter cycle: predict, update, resample, and state estimation.
        
        Parameters:
            measurement (np.ndarray): The observed measurement.
        
        Returns:
            np.ndarray: The estimated state.
        """
        self.predict()
        self.update(measurement)
        return self.state

    def run_filter(self, measurements):
        """
        This method is required for parameter estimation routines. It should run the filter 
        over a sequence of measurements and return both the state estimates and the residuals.
        
        Parameters:
            measurements (da.Array): Array of measurements over time, shape (n_timesteps, n_obs).
        
        Returns:
            state_estimates (da.Array): Filtered state estimates, shape (n_timesteps, n_state).
            residuals (da.Array): Residuals (measurement - predicted_measurement), same shape as measurements.
        
        For this simple example, we run the filter sequentially over the measurements.
        """
        n_timesteps = measurements.shape[0]
        state_estimates = []
        residuals = []
        for i in range(n_timesteps):
            meas = measurements[i]
            # Predict and update for current measurement
            self.step(meas)
            state_estimates.append(self.state)
            # Compute predicted measurement from the current state estimate
            if self.use_dask:
                pred_meas = da.dot(da.from_array(self.state, chunks=self.state.shape), self.observation_model.T).compute()
            else:
                pred_meas = self.observation_model @ self.state
            # Residual is difference between actual measurement and predicted measurement
            res = (meas.compute() if self.use_dask else meas) - pred_meas
            residuals.append(res)
        # Convert lists to dask arrays (or numpy arrays)
        state_estimates = da.from_array(np.vstack(state_estimates)) if self.use_dask else np.vstack(state_estimates)
        residuals = da.from_array(np.vstack(residuals)) if self.use_dask else np.vstack(residuals)
        return state_estimates, residuals

# Example usage:
if __name__ == '__main__':
    # Define a simple 2D state model (position and velocity)
    F = np.array([[1, 1],
                  [0, 1]])
    H = np.array([[1, 0]])  # Only position is measured

    # Define noise covariances
    Q = np.eye(2) * 0.01
    R = np.eye(1) * 0.1

    initial_state = np.array([0, 1])

    # Create the particle filter instance with Dask enabled
    pf = DaskParticleFilter(F, H, Q, R, initial_state, num_particles=1000, use_dask=True, estimation_strategy="residual_analysis")

    # Simulate some measurements over 10 time steps
    true_state = initial_state.copy()
    measurements_list = []
    np.random.seed(42)
    for _ in range(10):
        true_state = F @ true_state
        measurement = H @ true_state + np.random.normal(0, np.sqrt(R[0, 0]))
        measurements_list.append(measurement)
    measurements_array = da.from_array(np.vstack(measurements_list), chunks=(5, H.shape[0]))

    # Run the filter over these measurements (for parameter estimation, run_filter is used)
    state_estimates, residuals = pf.run_filter(measurements_array)
    print("Final estimated state:", pf.state)

    # Use one of the parameter estimation methods (e.g., residual_analysis)
    Q_est, R_est = pf.estimate_parameters(measurements_array)
    print("Estimated Q:", Q_est.compute() if pf.use_dask else Q_est)
    print("Estimated R:", R_est.compute() if pf.use_dask else R_est)
