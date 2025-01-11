class BaseEstimator:
    """
    Base class for all estimators in the TFilterPy package.
    Provides common functionality such as parameter handling and validation.
    """

    def __init__(self, name=None):
        """
        Initialize the BaseEstimator.

        Args:
            name (str): Optional name for the estimator.
        """
        self.name = name or self.__class__.__name__

    def get_params(self, deep=True):
        """
        Get parameters of the estimator.

        Args:
            deep (bool): If True, retrieves parameters of nested objects.

        Returns:
            dict: A dictionary of parameter names mapped to their values.
        """
        params = {}
        for key, value in self.__dict__.items():
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                params.update({f"{key}__{k}": v for k, v in deep_items})
            else:
                params[key] = value
        return params

    def set_params(self, **params):
        """
        Set parameters of the estimator.

        Args:
            **params: Arbitrary keyword arguments of parameters to set.

        Returns:
            self: Returns the instance itself.
        """
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter: {key}")
            setattr(self, key, value)
        return self

    def validate_matrices(self, matrices):
        """
        Validate that matrices have consistent shapes.

        Args:
            matrices (dict): A dictionary of matrix names and their values.

        Raises:
            ValueError: If the matrices are inconsistent.
        """
        for name, matrix in matrices.items():
            if not isinstance(matrix, (np.ndarray, da.Array)):
                raise ValueError(f"{name} must be a NumPy or Dask array.")

    def __repr__(self):
        """
        String representation of the estimator.

        Returns:
            str: A string representation of the estimator.
        """
        return f"{self.name}({self.get_params(deep=False)})"
