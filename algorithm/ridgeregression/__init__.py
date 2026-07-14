"""
AIMS-SDT Ridge Regression Infrastructure.

This package provides the core L2-regularized linear modeling capabilities 
for the AIMS-SDT tool. It is used as the foundational fitting engine for 
several trending strategies, including Polynomial and Fourier Expansion models.

Key Features:
*   **Regularized Optimization**: Solves the Ridge Regression equation 
    (X'X + αI)w = X'y using robust SVD solvers (:mod:`.ridge_regression`).
*   **Automatic Un-scaling**: Automatically transforms coefficients from 
    normalized internal units back to physical telemetry units for prediction.
*   **Training Framework**: Defines the abstract :class:`.ridge_reg_training_worker` 
    which manages data preparation, downsampling, and fitting for subclassed 
    linear models.
*   **Numerical Stability**: Integrates preprocessing pipelines (e.g., 
    StandardScaler) to handle ill-conditioned feature matrices.
"""
