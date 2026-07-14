"""
AIMS-SDT Time-based Polynomial Training Algorithm.

This sub-package provides implementations for modeling telemetry trends that 
are explicitly dependent on absolute time. It is effective for capturing 
long-term, non-periodic drift patterns.

Key features:
*   **Vandermonde Design Matrix**: Automatically constructs the high-order 
    expansion of the time variable for linear modeling (:mod:`.time_poly_trend_worker`).
*   **Efficient Least Squares**: Uses optimized singular value decomposition 
    (np.linalg.lstsq) to determine polynomial coefficients.
*   **Flexible Order and Indexing**: Supports polynomial models of any degree 
    and customizable starting powers as defined in algorithm attributes 
    (:mod:`.time_poly_trend`).
*   **Time-Normalized Predictions**: Performs high-fidelity trend generation 
    by mapping absolute Unix time to model-relative offsets.
*   **Pluggable Architecture**: The :class:`.time_poly_algorithm_factory` 
    provides the standardized entry point for this strategy.
"""
