"""
SDT Telemetry Visualization.

This package provides the tools for generating interactive graphical 
representations of telemetry data, trained models, and detected anomalies. 
It leverages Matplotlib for high-quality plotting and uses multiprocessing 
to ensure the UI remains responsive without blocking the main analysis engine.

Key functionalities include:
*   **Trend Visualization**: Overlaying trained predictive models on raw 
    telemetry streams to visually validate model fit (:mod:`.data_plot`).
*   **Anomaly Inspection**: Dedicated plots for highlighting detected outliers 
    and their statistical thresholds (:mod:`.handle_outlier_plot`).
*   **Interactive Annotations**: Tools for hovering over data points to 
    view precise timestamps and values.
*   **Multi-process Rendering**: Automatic spawning of independent GUI 
    processes for each requested plot window.
"""
