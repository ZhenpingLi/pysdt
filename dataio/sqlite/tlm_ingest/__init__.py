"""
SDT Telemetry Ingestion for SQLite.

This sub-package provides the necessary components for ingesting raw telemetry 
data into the SQLite database. It handles the conversion of raw telemetry 
data points into a format suitable for efficient storage, including data 
compression and batch processing.

Key components include:
*   :class:`.SQLiteDataIngestThread`: Manages the actual writing of data to SQLite.
*   :class:`.SQLiteIngestPacket`: A data structure for batching telemetry data.
*   :class:`.TlmData`: A basic data structure for individual telemetry points.
*   :class:`.TlmSQLiteDataConverter`: Orchestrates the conversion and ingestion process.
"""
