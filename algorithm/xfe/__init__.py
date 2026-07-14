"""
AIMS-SDT Ridge Extended Fourier Expansion (RidgeXFE) Algorithm.

This sub-package provides implementations for modeling complex periodic 
telemetry patterns that exhibit drift or change in amplitude over time. 
It extends the Fourier series concept by incorporating linear drift and 
time-modulated harmonic components.

Key features:
*   **Complex Harmonic Modeling**: Combines standard Fourier series with 
    linear trends and modulated harmonics to capture evolving patterns 
    (:mod:`.efe_function`).
*   **High-Dimensional Optimization**: Employs Ridge Regression with SVD 
    decomposition (np.linalg.lstsq) to accurately solve for a large 
    number of model parameters (:mod:`.ridge_xfe_training_worker`).
*   **Absolute Relative Prediction**: Utilizes normalized time relative to 
    an absolute reference epoch to enable drift modeling across multiple 
    cycles (:mod:`.xfe_trend`).
*   **Pluggable Architecture**: The :class:`.ridge_xfe_algorithm_factory` 
    provides the standardized entry point for this advanced periodic strategy.
"""
