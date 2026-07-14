"""
SDT Database Access Layer.

This package provides a structured interface to the satellite telemetry database. 
It handles the loading and parsing of satellite-specific JSON configuration 
files, maps mnemonics to their physical subsystems, and manages algorithm 
definitions and operational state configurations.

The core module, :mod:`sdt_db`, provides singleton-style access to the 
underlying data models and manages an internal cache for high-performance 
lookups during training sessions.
"""
