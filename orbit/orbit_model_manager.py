import logging
import os
import sys
from typing import List, Optional

import numpy as np

from orbit.geo_orbit_time import GEOOrbitTime
from orbit.leo_orbit_time import LEOOrbitTime
from orbit.long_term_trend_time import LongTermTrendTime
from training import data_buffer
from training.training_set import TrainingSet

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import sdt_config
from orbit.orbit_if import OrbitIF
from orbit.model_time import ModelTime
from algorithm.algorithm_def import AlgorithmDef


# Module-level variables to act as static fields
_orbit_if: Optional[OrbitIF] = None
_l_orbit: Optional[OrbitIF] = None
CONTEXT = "ORBITMODEL"

# Constants
LONGTERM = 1


def _init_orbit_if():
    """
    Initializes the active OrbitIF implementation based on current configuration.
    
    It checks the 'session_type' and 'ORBITTYPE' configuration to determine 
    whether to use LongTermTrendTime, LEOOrbitTime, or GEOOrbitTime.
    """
    global _orbit_if
    
    session_type = data_buffer.session_type
    
    if session_type == LONGTERM:
        _orbit_if = LongTermTrendTime()
        return

    orbit_type = sdt_config.get_config_value("ORBITTYPE")
    if orbit_type == "LEO":
        _orbit_if = LEOOrbitTime()
    elif orbit_type == "GEO":
        _orbit_if = GEOOrbitTime()
    else:
        logging.error(f"{CONTEXT}: Unrecognized orbit type: {orbit_type}")
        _orbit_if = None

def create_default_model_time(mnemonic_id: str) -> Optional[ModelTime]:
    """
    Constructs the default temporal context for a model in the current session.

    Args:
        mnemonic_id (str): ID of the telemetry mnemonic.

    Returns:
        Optional[ModelTime]: The default time configuration, or None if OrbitIF 
            is uninitialized.
    """
    global _orbit_if
    if _orbit_if is None:
        _init_orbit_if()
    
    if _orbit_if:
        return _orbit_if.get_default_model_time(mnemonic_id)
    return None

def create_default_model_time_for_prev_session(mnemonic_id: str) -> Optional[ModelTime]:
    """
    Constructs the default temporal context for the previous training session.
    Used for incremental model updates.

    Args:
        mnemonic_id (str): ID of the mnemonic.

    Returns:
        Optional[ModelTime]: The time configuration from the previous session.
    """
    global _orbit_if
    if _orbit_if is None:
        _init_orbit_if()
        
    if _orbit_if:
        return _orbit_if.get_default_model_time_for_prev_session(mnemonic_id)
    return None

def get_model_time(alg: AlgorithmDef, frequency: float, training_set: TrainingSet) -> Optional[ModelTime]:
    """
    Determines the refined model time parameters based on the specific 
    algorithm needs and available data points.

    Args:
        alg (AlgorithmDef): The algorithm definition.
        frequency (float): Data sampling frequency.
        training_set (TrainingSet): The actual data points for the session.

    Returns:
        Optional[ModelTime]: The calculated time parameters.
    """
    global _orbit_if
    if _orbit_if is None:
        _init_orbit_if()
        
    if _orbit_if:
        return _orbit_if.get_model_time(alg, frequency, training_set)
    return None

def get_input_trend_times(pattern_period: float, session_time: float, num_pattern_in_training: int, is_orbitbased: bool) -> Optional[np.ndarray]:
    """
    Calculates the sequence of timestamps defining the training window 
    for trend analysis.

    Args:
        pattern_period (float): Single cycle duration in seconds.
        session_time (float): Current session reference timestamp.
        num_pattern_in_training (int): Number of cycles to include.
        is_orbitbased (bool): If True, aligns window with orbital events.

    Returns:
        Optional[np.ndarray]: Array of start/end timestamps.
    """
    global _orbit_if
    if _orbit_if is None:
        _init_orbit_if()
        
    if _orbit_if:
        return _orbit_if.get_input_trend_times(pattern_period, session_time, num_pattern_in_training, is_orbitbased)
    return None

def get_session_time() -> Optional[np.ndarray]:
    """
    Retrieves the start and end timestamps for the active training session.

    Returns:
        Optional[np.ndarray]: Array [start, end] in Unix seconds.
    """
    global _orbit_if
    if _orbit_if is None:
        _init_orbit_if()
        
    if _orbit_if:
        return _orbit_if.get_session_time()
    return None

def get_lt_session_time() -> Optional[np.ndarray]:
    """
    Retrieves the start and end timestamps for a long-term session context.

    Returns:
        Optional[np.ndarray]: Array [start, end].
    """
    global _l_orbit
    if _l_orbit is None:
        _l_orbit = LongTermTrendTime()
        
    return _l_orbit.get_session_time()
