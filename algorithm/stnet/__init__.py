"""
AIMS-SDT State-based Neural Network (STNET).

This sub-package provides implementations for modeling telemetry during 
specific operational states (e.g., maneuvers, eclipses) using Neural 
Networks. It is designed to capture the unique patterns and dynamics 
associated with different phases of satellite operation.

Key features:
*   **State-Specific Topology**: Automatically re-configures neural network 
    hidden layer sizes based on state-specific dimensionality requirements 
    (:mod:`.state_nn_trend`).
*   **Persistent Weight Management**: Implements a local archival system for 
    storing and retrieving optimized network weights to enable high-fidelity 
    incremental fitting (:mod:`.state_nn_trend_param_io`).
*   **Dedicated Normalization**: Employs specialized preprocessing to 
    standardize telemetry within state boundaries (:mod:`.state_pre_processing`).
*   **Integrated Fitting Pipeline**: Orchestrates the multi-stage training 
    process including parameter retrieval, downsampling, and L-BFGS 
    optimization (:mod:`.state_nn_training_worker`).
*   **Pluggable Integration**: The :class:`.state_nn_algorithm_factory` 
    provides the standardized entry point for this state-aware modeling strategy.
"""
