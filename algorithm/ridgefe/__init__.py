"""
AIMS-SDT Ridge Fourier Expansion (RidgeFE) Algorithm.

This sub-package implements models for periodic telemetry patterns using 
Fourier series expansion combined with Ridge Regression (L2 regularization). 
This approach is highly effective for capturing multi-harmonic behaviors 
(e.g., thermal cycles) while preventing overfitting in high-order models.

Key features:
*   **Harmonic Modeling**: Automatically generates sine and cosine basis 
    functions for multiple harmonic frequencies (:mod:`.fourier_expansion_function`).
*   **Regularized Fitting**: Employs Ridge Regression to handle potentially 
    ill-conditioned feature matrices and improve generalization (:mod:`.ridge_fe_training_worker`).
*   **Orbit-Relative Predictions**: Maps learned Fourier coefficients back to 
    physical time within orbital cycles for trend generation (:mod:`.ridge_fe_trend`).
*   **Redundancy Filtering**: Includes utilities to detect and remove collinear 
    periodic features to ensure numerical stability.
*   **Pluggable Architecture**: The :class:`.ridge_fe_algorithm_factory` 
    provides the standardized entry point for this strategy.
"""
