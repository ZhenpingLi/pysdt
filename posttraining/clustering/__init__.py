"""
SDT Event Clustering and Analysis.

This sub-package implements the algorithms and data structures for identifying 
meaningful events from telemetry outliers. It uses a density-based approach 
(DBSCAN) to group related anomalies into hierarchical event structures.

Key Features:
*   **Hierarchical Representation**: Events are modeled as trees, aggregating 
    from individual mnemonics up to subsystems and the entire satellite (:mod:`.sdt_event`).
*   **Similarity Analysis**: Grouping is based on a custom scalar product 
    metric that assesses the correlation between different telemetry 
    outlier patterns (:mod:`.h_cluster`).
*   **Clustering Orchestration**: Manages the integration of current session 
    events with historical data to identify recurring issues vs. random noise (:mod:`.event_clustering`).
*   **Adaptive Filtering**: Clusters are refined based on temporal density and 
    occurrence frequency across multiple days.
"""
