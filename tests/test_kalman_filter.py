import numpy as np
import pytest

from TFilterPy.state_estimation.linear_filters import DaskKalmanFilter


def test_kalman_filter_initialization():
    F = np.eye(2)
    H = np.eye(2)
    Q = np.eye(2)
    R = np.eye(2)
    x0 = np.zeros(2)
    P0 = np.eye(2)

    # Should initialize without error
    kf = DaskKalmanFilter(F, H, Q, R, x0, P0)
    assert kf.F.shape == (2, 2)

    # Test non-square F matrix
    with pytest.raises(ValueError):
        DaskKalmanFilter(np.array([[1, 0]]), H, Q, R, x0, P0)


def test_kalman_predict():
    F = np.eye(2)
    H = np.eye(2)
    Q = np.eye(2) * 0.1
    R = np.eye(2) * 0.5
    x0 = np.zeros(2)
    P0 = np.eye(2)
    measurements = np.random.randn(100, 2)  # 100 time steps

    kf = DaskKalmanFilter(F, H, Q, R, x0, P0)
    kf.fit(measurements)
    state_estimates = kf.predict().compute()

    assert state_estimates.shape == (100, 2)


# def test_dask_compatibility():
#     F = np.eye(2)
#     H = np.eye(2)
#     Q = np.eye(2)
#     R = np.eye(2)
#     x0 = np.zeros(2)
#     P0 = np.eye(2)
#     measurements = da.random.randint(2, 300, chunks=(50, 2))

#     kf = DaskKalmanFilter(F, H, Q, R, x0, P0)
#     kf.fit(measurements)
#     state_estimates = kf.predict()

#     assert isinstance(state_estimates, da.Array)
#     assert state_estimates.compute().shape == (200, 2)
