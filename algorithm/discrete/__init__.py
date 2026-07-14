"""
AIMS-SDT 'Discrete' Training Algorithm.

This sub-package provides components specialized for modeling telemetry
that jumps between discrete, quantized levels (e.g., state indicators,
raw counts).

Key features:
*   **Characteristic Delta Identification**: Automatically detects the typical
    distance between discrete levels to prevent false positive outlier
    detections (:mod:`.value_counter`).
*   **Floor Sigma Logic**: Enforces a minimum standard deviation floor based
    on the identified delta, ensuring the model is robust to quantization noise
    (:mod:`.discrete_training_worker`).
*   **Hybrid Modeling**: Uses statistical analysis for baseline fitting while
    adapting to discrete value distributions.
*   **Standard Prediction**: Leverages :class:`.default_trend.DefaultTrend` for
    predictive evaluation.
"""
