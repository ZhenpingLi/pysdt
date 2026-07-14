"""
SDT 'Default' Training Algorithm.

This sub-package provides the implementation for the 'default' baseline 
training algorithm. It is used for telemetry that is generally constant 
over time or contains random Gaussian noise.

Key components:
*   :class:`.default_training_worker.DefaultTrainingWorker`: Performs 
    iterative statistical analysis (mean/sigma calculation).
*   :class:`.default_trend.DefaultTrend`: Predictive model that uses the 
    learned mean as a constant baseline.
*   :class:`.default_algorithm_factory.DefaultAlgorithmFactory`: The entry 
    point for the PluginManager to load the default algorithm components.

This algorithm uses a fixed 48-hour session window with two 24-hour pattern 
cycles for its statistical baseline.
"""
