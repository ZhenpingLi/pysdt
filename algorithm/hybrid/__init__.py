"""
AIMS-SDT 'Hybrid' Multi-State Training Algorithm.

This sub-package provides a sophisticated implementation for modeling 
telemetry systems defined by Finite State Machines (FSM). It allows for 
different trending algorithms to be active during specific operational 
states (e.g., modeling 'ECLIPSE' with a Neural Network while using 
'DEFAULT' statistics for 'NORMAL' operations).

Key features:
*   **State-Based Segmentation**: Automatically splits session telemetry 
    into discrete training sets based on active operational zones (:mod:`.h_pre_processing`).
*   **Composite Model Coordination**: Manages a collection of sub-trends, 
    efficiently identifying which model is active at any given timestamp 
    (:mod:`.hybrid_trend`).
*   **Parallel-Style Fitting**: Orchestrates the execution of specialized 
    sub-workers for each identified state segment (:mod:`.hybrid_training_worker`).
*   **Dynamic Boundary Refinement**: Adjusts state zone transitions based on 
    telemetry features to ensure high model fidelity.
*   **Pluggable Architecture**: The :class:`.hybrid_algorithm_factory` provides 
    the entry point for integrating multi-state modeling into the system.
"""
