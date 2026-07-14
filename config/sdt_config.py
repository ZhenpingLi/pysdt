from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Any

import numpy as np

from util.time_util import get_time_from_string, HOUR_IN_SECONDS

# Determine the path to the config.json file relative to this script
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
config_path = os.path.join(parent_dir, 'sdt-config','config.json')
PYSDT_HOME = Path.home() / ".pysdt"
APP_JSON_PATH = PYSDT_HOME / "application.json"
_config = {}

# Global state variables
sat_id: Optional[str] = None
debug: str = "off"
config_dir: Optional[str] = None

# Configuration parameters with default values
TPCWARNING = 2.0
TPCERROR = 4.0
OCWARNING = 1.0
OCERROR = 2.0
ECLDUR = 0.0
GAPLIMIT = 0.5
BATCHNUM = 4
SAMPLETIME = 2.0
warning_limits: Optional[np.ndarray] = None
error_limits: Optional[np.ndarray] = None
session_period = 48.0 * 3600
num_pattern_in_training = 1
mission_start: Optional[float] = None
training_delta = 24.0 * 3600
cluster_threshold = 5.0
event_period_limit = 3600.0


def set_sat_id(new_sat_id: str):
    """
    Sets the active satellite ID and initializes its specific configuration.

    It looks up the configuration directory from application.json and
    triggers the loading of the satellite-specific config.json.

    Args:
        new_sat_id (str): The satellite identifier (e.g., 'G19').
    """
    global sat_id, config_dir
    sat_id = new_sat_id
    config_dir = get_config_path()
    load_config(config_dir)

def load_config(_config_dir: str):
    """
    Loads configuration settings from a JSON file.

    Args:
        _config_dir (str): The directory containing the config.json file.
            If None, uses the project root default.
    """
    global _config, config_dir
    if _config_dir is not None:
        config_dir = _config_dir
        _config_path = os.path.join(config_dir, 'config.json')
    else:  # use default
        _config_path = config_path
        
    if os.path.exists(_config_path):
        try:
            with open(_config_path, 'r') as f:
                _config = json.load(f)
                _init_config_values()
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {_config_path}: {e}")
    else:
        print(f"Warning: Config file not found at {_config_path}")


def _init_config_values():
    """
    Initializes global numeric and time-based configuration constants from
    the loaded dictionary. Converts strings to appropriate numeric types.
    """
    global TPCWARNING, TPCERROR, OCWARNING, OCERROR, ECLDUR, GAPLIMIT, BATCHNUM, SAMPLETIME, warning_limits, error_limits
    TPCWARNING = get_float_property("TCWARNING") or 2.0
    TPCERROR = get_float_property("TCERROR") or 4.0
    OCWARNING = get_float_property("OUTLIERWARNINGLIMIT") or 1.0
    OCERROR = get_float_property("OUTLIERERRORLIMIT") or 2.0
    ECLDUR = get_float_property("ECLDUR") or 0.0
    GAPLIMIT = get_float_property("GAPTHRESHOLD") or 0.5
    BATCHNUM = get_int_property("BATCHNUM") or 4
    SAMPLETIME = get_float_property("SAMPLETIME")
    warning_limits = np.array([TPCWARNING, OCWARNING])
    error_limits = np.array([TPCERROR, OCERROR])
    
    global session_period, num_pattern_in_training, mission_start, training_delta, cluster_threshold, event_period_limit
    session_period = get_float_property("SESSIONPERIOD") * HOUR_IN_SECONDS if get_float_property("SESSIONPERIOD") else 48.0 * HOUR_IN_SECONDS
    num_pattern_in_training = get_int_property("NUMPATTERNINTRAINING") or 1
    mission_start = get_time_from_string(get_string_property("MISSIONSTART"))
    training_delta = get_float_property("TRAININGDELTA") * HOUR_IN_SECONDS if get_float_property("TRAININGDELTA") else 24.0 * HOUR_IN_SECONDS
    cluster_threshold = float(get_config_value("CLUSTERTHRESHOLD") or 5.0)
    event_period_limit = float(get_config_value("EVENTTIMELIMIT") or 3600.0)

def get_config_value(key: str) -> Optional[Any]:
    """
    Retrieves a raw value from the configuration dictionary.

    Args:
        key (str): The key to look up. Supports nested keys using '/'
            (e.g., 'INFLUXDB/ORG').

    Returns:
        Optional[Any]: The value associated with the key, or None if not found.
    """
    keys = key.split('/')
    value = _config
    try:
        for k in keys:
            value = value[k]
        return value
    except (KeyError, TypeError):
        logging.error(f"Key '{key}' not found in configuration.")
        return None


def get_int_property(prop_name: str) -> Optional[int]:
    """
    Retrieves an integer property.

    Args:
        prop_name (str): The configuration key.

    Returns:
        Optional[int]: The integer value, or None if invalid or missing.
    """
    val = get_config_value(prop_name)
    if val is not None:
        try:
            return int(val)
        except ValueError:
            return None
    return None


def get_string_property(prop_name: str) -> Optional[str]:
    """
    Retrieves a string property. If a list is found, returns its first element.

    Args:
        prop_name (str): The configuration key.

    Returns:
        Optional[str]: The string value, or None if missing.
    """
    val = get_config_value(prop_name)
    if val is not None:
        if isinstance(val, list):
            if len(val) > 0:
                return str(val[0])
            return str(val)
        return str(val)
    return None


def get_float_property(prop_name: str) -> Optional[float]:
    """
    Retrieves a float property.

    Args:
        prop_name (str): The configuration key.

    Returns:
        Optional[float]: The float value, or None if invalid or missing.
    """
    val = get_config_value(prop_name)
    if val is not None:
        try:
            return float(val)
        except ValueError:
            logging.error(f"Error converting configuration key '{prop_name}' value '{val}' to float")
            return None
    return None


def get_list_property(prop_name: str) -> Optional[List[str]]:
    """
    Retrieves a list of strings from the configuration.

    Args:
        prop_name (str): The configuration key.

    Returns:
        Optional[List[str]]: The list, or None if not a list or missing.
    """
    val = get_config_value(prop_name)
    if val is not None and isinstance(val, list):
        return [str(item) for item in val]
    return None


def get_string_from_list_by_index(prop_name: str, index: int) -> Optional[str]:
    """
    Retrieves a specific string from a configuration list by its index.

    Args:
        prop_name (str): The key for the list property.
        index (int): The index to retrieve.

    Returns:
        Optional[str]: The string at the index, or None if out of bounds or invalid.
    """
    val = get_config_value(prop_name)
    if val is not None and isinstance(val, list):
        if 0 <= index < len(val):
            return str(val[index])
    return None

def get_config_path() -> Optional[str]:
    """
    Determines the satellite-specific configuration path by reading
    the application.json registry.

    Returns:
        Optional[str]: The path to the satellite configuration directory,
            or None if mapping is missing.
    """
    if os.path.exists(APP_JSON_PATH):
        try:
            with open(APP_JSON_PATH, 'r') as f:
                mapping = json.load(f)
                return mapping.get(sat_id.lower())
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {APP_JSON_PATH}: {e}")
    else:
        print(f"Warning: application.json not found at {APP_JSON_PATH}")
    return None
