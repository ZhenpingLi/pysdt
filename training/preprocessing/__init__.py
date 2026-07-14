"""
SDT Telemetry Data Preprocessing.

This package provides a comprehensive suite of tools for preparing raw satellite 
telemetry for data training. It handles the critical steps of data cleaning, 
statistical analysis, and feature engineering.

Key functionalities include:
*   **Statistical Analysis**: Per-cycle calculation of mean, max, min, and 
    standard deviation (:mod:`.data_stat`).
*   **Exclusion Zones**: Identification and management of time regions that 
    should be ignored (e.g., maneuvers, eclipses) (:mod:`.ex_zone`).
*   **Data Normalization**: Orbit-aware transformations using various scaling 
    strategies (MinMax, Max, Min, Uniform) (:mod:`.orbit_based_transform`, :mod:`.coefficient_inf`).
*   **Feature Engineering**: Generation of derivative and mixed-type input 
    features (:mod:`.data_model_utility`).
*   **Downsampling**: Shape-preserving data reduction using the LTTB algorithm (:mod:`.lttb_filter`).
*   **State Management**: Definition of temporal boundaries for finite state 
    machine models (:mod:`.state_zones`).

The package implements distinct strategies for :mod:`.short_term_processing` 
and :mod:`.long_term_processing` sessions.
"""
