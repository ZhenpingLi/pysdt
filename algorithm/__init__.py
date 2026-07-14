"""
SDT Training and Trending Algorithms.

This package contains the core analytical components for telemetry modeling. 
It implements a pluggable architecture for various training algorithms, 
including:

*   **Neural Networks**: Feed-forward models with L-BFGS optimization (:mod:`.fbnn`).
*   **Ridge Regression**: Including Fourier Expansion (MFE) for periodic data (:mod:`.ridgeregression`).
*   **Polynomial Trending**: For linear and low-order non-linear drift.

The package also defines the fundamental data structures used across all 
algorithms, such as :class:`.DataTrend`, :class:`.TrendNode`, and :class:`.Outlier`.
"""
