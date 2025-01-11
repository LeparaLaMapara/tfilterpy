<p align="center">
  <img src="branding/logo/tfilters-logo.jpeg?" alt="tfilterspy logo"/>
</p>

# **TFilterPy** 🌀

![PyPI Version](https://img.shields.io/pypi/v/tfilterpy?color=blue&label=PyPI&style=for-the-badge)
![Tests](https://github.com/LeparaLaMapara/tfilterpy/actions/workflows/python-tests.yml/badge.svg?style=for-the-badge)
![Build](https://github.com/LeparaLaMapara/tfilterpy/actions/workflows/publish.yml/badge.svg?style=for-the-badge)
![License](https://img.shields.io/github/license/LeparaLaMapara/tfilterpy?color=green&style=for-the-badge)

✨ **TFilterPy** is your new favorite Python library for implementing state-of-the-art Bayesian filtering techniques like Kalman Filters and Particle Filters. Whether you're working on noisy linear systems, nonlinear dynamics, or want to sound cool at a party when you say "I coded my own Kalman Filter," this is the library for you!

---

## **What’s Inside?** 📦

🎉 **TFilterPy** offers:
- **Kalman Filters** 🧮 – A classic but still iconic tool for linear filtering and smoothing.
- **Particle Filters** 🎲 – Sampling-based estimators for nonlinear systems.
- **Nonlinear Filters** 🔀 – For when your system decides to be complicated.
- Extensible design for implementing more advanced filtering algorithms like Unscented Kalman Filters (UKF) and beyond.

---

## **Installation** 🚀

Getting started is as easy as pie (or Pi)! 🍰

```bash
pip install tfilterpy
```

Want to contribute or tinker with the code? Clone the repo and install the development dependencies:

```bash
git clone https://github.com/LeparaLaMapara/tfilterpy.git
cd tfilterpy
pip install .[dev]
```
___________________________________________

## Usage 🛠️
Example 1: Using a Kalman Filter to tame noisy data 🤖

```python
import numpy as np
from TFilterPy.state_estimation.linear_filters import DaskKalmanFilter

# Define your system
F = np.eye(2)
H = np.eye(2)
Q = np.eye(2) * 0.01
R = np.eye(2) * 0.1
x0 = np.zeros(2)
P0 = np.eye(2)

# Create a Kalman Filter
kf = DaskKalmanFilter(F, H, Q, R, x0, P0)

# Simulate some noisy measurements
measurements = np.random.randn(100, 2)

# Run the filter
filtered_states = kf.run_filter(measurements)
print(filtered_states.compute())
```


_____________________
## Features 🌟

  - Dask Support for large-scale filtering with parallelism 🏎️
  - Modular structure for extensibility 🛠️
  - Lightweight and easy to use 👌
  - Designed for both linear and nonlinear systems 🔄

___________________________________
# Why TFilterPy? 💡

Because Kalman deserves better branding! Instead of grappling with matrices and equations from scratch, use TFilterPy and focus on the fun part: tweaking models until they (hopefully) work. 🎉
______________________________


## Contributing 🤝

We welcome contributions of all types:

  - 🐛 Found a bug? Let us know in the Issues.
  - 🌟 Want to add a feature? Fork the repo, make your changes, and create a pull request.
  - 🧪 Testers needed! Write more test cases for improved coverage.

### Development Setup
  1. Clone the repo:
  ```bash
    git clone https://github.com/LeparaLaMapara/tfilterpy.git
  ```
  2. Install dependencies:
  ```bash
    pip install .[dev]
  ```
  3. Run tests:
  ```bash
    pytest tests/
  ```

  
  _________________________
## Future Plans 🔮

  - Adding Unscented Kalman Filters (UKF) 🦄
  - Implementing Gaussian Process Filters 📈
  - Enhancing scalability with advanced parallelism ⚡

________________

## Documentation 📚

Detailed documentation is available at: https://leparalamapara.github.io/tfilterpy
(Yes, we made it look fancy. You're welcome. ✨)
_____________________

## Support ❤️

If this library made your life easier, consider:

    Giving it a ⭐ on GitHub.
    Telling your friends, colleagues, and cats about TFilterPy.
_________________________

## License 📜

This project is licensed under the MIT License. Feel free to use it, modify it, or use it as a coaster.

**Enjoy your filtering adventures with TFilterPy! 🎉🚀**