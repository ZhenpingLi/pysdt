"""
SDT Data Training and Session Management.

This package orchestrates the entire data training workflow, from session 
initialization and data retrieval to algorithm execution and post-training 
analysis. It provides the framework for processing telemetry data to build 
and refine predictive models.

Key functionalities include:
*   **Session Management**: Defines and manages the context of a training 
    session, including time boundaries, satellite ID, and global caches (:mod:`.data_buffer`).
*   **Training Orchestration**: Coordinates the execution of training tasks, 
    often leveraging parallel processing (e.g., Ray) for efficiency (:mod:`.sdt_training_session`).
*   **Algorithm Execution**: Provides an abstract interface (:class:`.training_worker.TrainingWorker`) 
    for pluggable training algorithms and manages their lifecycle.
*   **Data Preprocessing**: Includes utilities for preparing raw telemetry 
    for training, such as handling data gaps, normalization, and exclusion zones (:mod:`.preprocessing`).
*   **Data Structures**: Defines core data containers like :class:`.training_set.TrainingSet` 
    for holding input/output data and metadata.
"""
