"""
SDT Data Input/Output.

This package provides the infrastructure for retrieving telemetry data from 
external sources and persisting training results to a data archive. 

It defines abstract interfaces (:class:`.SDTDataInput` and :class:`.SDTDataTrainingIO`) 
to allow for pluggable backends. Currently supported implementations include:

*   **SQLite**: A high-performance local archive using Zstandard compression 
    for telemetry and binary BLOBs for model parameters.
*   **InfluxDB**: (Interface defined, implementations can be added).

The package also includes utility components for data ingestion and mapping 
between mnemonic names and integer IDs for optimized storage.
"""
