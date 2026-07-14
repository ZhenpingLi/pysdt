"""
SDT Post-Training Analysis and Monitoring.

This package provides the tools and framework for analyzing the outputs of 
data training sessions to determine the operational health and safety status 
of satellite telemetry.

Key functionalities include:
*   **Data Quality Metrics**: Evaluation of various metrics (e.g., Temporal 
    Change, Outlier Cluster Value) to assess mnemonic health (:mod:`.data_quality_metrics`).
*   **Event Clustering**: Grouping of related outliers into meaningful events 
    using density-based algorithms (:mod:`.clustering`).
*   **Status Reporting**: Generation of comprehensive status reports for 
    individual mnemonics and entire subsystems (:mod:`.mnemonic_status`).
*   **Analysis Orchestration**: A high-level manager (:mod:`.post_training_processing_manager`) 
    that coordinates the entire post-training workflow.

The package defines an abstract :class:`.Analyzer` interface, with concrete 
implementations like :class:`.sdt_analyzer` providing the core logic.
"""
