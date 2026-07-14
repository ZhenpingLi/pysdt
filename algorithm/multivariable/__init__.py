"""
SDT Multi-Variable Polynomial Training Algorithm.

This sub-package implements models that describe a primary telemetry mnemonic 
as a function of multiple independent variables (e.g., time plus other 
telemetry points).

Key features:
*   **Dynamic Dimensionality**: Supports any number of input variables as 
    defined in the algorithm configuration.
*   **Non-Linear Least Squares**: Uses high-performance optimization 
    (via scipy's curve_fit) to solve for multi-variable coefficients 
    (:mod:`.multi_variable_trend_worker`).
*   **Flexible Feature Construction**: Automatically builds polynomial 
    feature vectors including bias and cross-variable terms (:mod:`.multi_variable_trend`).
*   **Composite Prediction**: Efficiently performs inference using dot 
    products between features and learned weights.
*   **Pluggable Integration**: The :class:`.multi_variable_algorithm_factory` 
    provides the standardized entry point for this strategy.
"""
