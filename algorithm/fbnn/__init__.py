"""
AIMS-SDT Feed-forward Back-propagation Neural Network (FBNN).

This sub-package provides a robust implementation for modeling complex, 
periodic, and non-linear telemetry trends using Multi-layer Perceptrons (MLP).

Key features:
*   **High-Precision Optimization**: Uses the L-BFGS quasi-Newton solver for 
    fast and accurate convergence on telemetry datasets (:mod:`.neural_net_wrapper`).
*   **Global Initialization**: Overcomes local minima by executing multiple 
    parallel random restarts and selecting the global best initialization 
    (:mod:`.nn_init_training`).
*   **Orbit-Aware Preprocessing**: Automatically normalizes time features 
    within detected orbital cycles to improve model learnability.
*   **Incremental Retraining**: Supports high-fidelity updates of existing 
    model weights (:mod:`.nn_training_worker`).
*   **Serialized Model Management**: Handles the encoding and decoding of 
    complex network weights for database storage (:mod:`.fbnn_trend`).
"""
