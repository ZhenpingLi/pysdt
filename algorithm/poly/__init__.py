"""
SDT 'Polynomial' Training Algorithm.

This sub-package provides implementations for modeling telemetry that exhibits 
linear or higher-order non-linear drift over time.

Key components:
*   **Robust Fitting**: Uses non-linear least squares optimization with 
    outlier-robust loss functions (soft_l1) (:mod:`.poly_trend_worker`).
*   **Flexible Order**: Supports polynomial models of any degree as defined 
    in the algorithm's dimensionality attributes (:mod:`.poly_trend`).
*   **Feature Expansion**: Automatically generates polynomial terms and bias 
    vectors for accurate modeling of gradual changes.
*   **Pluggable Architecture**: The :class:`.poly_trend_algorithm_factory` 
    provides the standardized entry point for this trending strategy.
"""
