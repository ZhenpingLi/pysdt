"""
SDT Orbital Time Modeling.

This package provides the temporal intelligence required for satellite 
telemetry analysis. It handles the mapping between absolute timestamps and 
physical orbital phases (e.g., ascending node crossings).

The package implements various models to accommodate different mission types:
*   **LEO Modeling**: Precise cycle detection using zero-crossing interpolation 
    of positional telemetry (:mod:`.leo_orbit_time`).
*   **GEO Modeling**: Simplified daily cycle alignment for geostationary 
    missions (:mod:`.geo_orbit_time`).
*   **Long-Term Trending**: Alignment with seasonal and yearly cycles.

A centralized :mod:`.orbit_model_manager` orchestrates the selection and 
initialization of these models based on satellite configuration.
"""
