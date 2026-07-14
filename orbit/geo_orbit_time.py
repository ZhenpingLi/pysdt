import os
import sys

import numpy as np

from training.training_set import TrainingSet

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import sdt_config
from algorithm.algorithm_def import AlgorithmDef
from orbit.model_time import ModelTime
from orbit.orbit_if import OrbitIF
from orbit.pattern_time_evaluate import PatternTimeEvaluate
from util.time_util import get_current_day_start, get_datetime

# --- Constants ---
DAY_IN_SECONDS = 86400
HOUR_IN_SECONDS = 3600
SHORTTERM = 0
FBNN = "fbnn"
CONTEXT = "GEOORBITTIME"


class GEOOrbitTime(OrbitIF):
    """
    OrbitIF implementation for Geostationary (GEO) satellites.
    
    In this model, the orbital period is assumed to be exactly one solar day 
    (24 hours). The class provides methods to set up training cycles aligned 
    with daily patterns.
    """

    def __init__(self):
        """Initializes the GEO orbit time evaluator."""
        self.evaluate = PatternTimeEvaluate()

    def get_default_model_time_for_prev_session(self, alg: AlgorithmDef) -> ModelTime:
        """
        Calculates the temporal context for the previous training session.
        
        This is used to retrieve baseline models for incremental training. It 
        calculates a window ending one pattern period before the current 
        session end.

        Args:
            alg (AlgorithmDef): The algorithm definition used to determine 
                the number of patterns required.

        Returns:
            ModelTime: The calculated time context for the previous session.
        """
        # Delayed import to avoid circular dependency
        from training.data_buffer import session_end

        np_val = max(1, alg.np)
            
        pattern_period = DAY_IN_SECONDS
        prev_session_end = session_end - pattern_period
        prev_session_start = prev_session_end - (2 * np_val) * pattern_period
        
        model_time = ModelTime(prev_session_start, prev_session_end)
        
        num_patterns = 2 * np_val
        # Define starting times for each 24-hour pattern cycle
        ref_times = prev_session_start + np.arange(num_patterns + 1) * DAY_IN_SECONDS
        
        model_time.set_pattern_times(ref_times)
        model_time.set_reference_time(prev_session_start)
        model_time.set_model_period(DAY_IN_SECONDS)
        
        return model_time

    def get_default_model_time(self, alg: AlgorithmDef) -> ModelTime:
        """
        Calculates the temporal context for the current training session.
        
        Aligned with daily cycles, it determines the session boundaries and 
        pattern start times based on the current session start and period.

        Args:
            alg (AlgorithmDef): The algorithm definition.

        Returns:
            ModelTime: The populated time context for the current session.
        """
        from training.data_buffer import session_start, session_end

        start = session_start
        session_period = sdt_config.session_period
        if alg.np >= 1:
            start = session_end - alg.np * session_period
            
        model_time = ModelTime(start, session_end)
        
        num_patterns = int(round((session_end - start) / DAY_IN_SECONDS))
        ref_times = start + np.arange(num_patterns + 1) * DAY_IN_SECONDS
        
        model_time.set_pattern_times(ref_times)
        model_time.set_reference_time(start)
        model_time.set_model_period(DAY_IN_SECONDS)
        
        return model_time

    def _update_start(self, alg_def: AlgorithmDef, start: float) -> float:
        """
        Adjusts the start time to align with the beginning of a UTC day for 
        specific algorithms like FBNN.

        Args:
            alg_def (AlgorithmDef): The algorithm definition.
            start (float): The original start timestamp.

        Returns:
            float: The adjusted start timestamp.
        """
        if alg_def.get_name() == FBNN:
            dt = get_datetime(start)
            hour = dt.hour
            if hour != 0:
                return start - hour * HOUR_IN_SECONDS
        return start

    def get_model_time(self, alg: AlgorithmDef, frequency: float, training_set: TrainingSet) -> ModelTime:
        """
        Refines the model time using data-driven evaluation.

        Args:
            alg (AlgorithmDef): The algorithm definition.
            frequency (float): Sampling frequency.
            training_set (TrainingSet): The session data.

        Returns:
            ModelTime: The evaluated time context.
        """
        return self.evaluate.get_model_time(alg, frequency, training_set)

    def get_input_trend_times(self, pattern_period: float, session_time: float, num_pattern_in_training: int, is_orbitbased: bool) -> np.ndarray:
        """
        Calculates the time range required for retrieving the baseline model.

        Args:
            pattern_period (float): The cycle duration (24h).
            session_time (float): Current session time.
            num_pattern_in_training (int): Cycles to include.
            is_orbitbased (bool): Unused in GEO implementation.

        Returns:
            np.ndarray: Array [start_time, window_size].
        """
        from training.data_buffer import get_training_delta
        
        input_times = np.zeros(2)
        training_delta = get_training_delta() * HOUR_IN_SECONDS
        
        # Calculate range ending at the previous session boundary
        input_times[0] = session_time - 2 * num_pattern_in_training * pattern_period - training_delta
        input_times[1] = training_delta / 2.0
        
        return input_times

    def get_session_time(self) -> np.ndarray:
        """
        Determines the default boundaries for a GEO training session.

        Returns:
            np.ndarray: Array [start, end] covering the configured session period 
                ending at the start of the current day.
        """
        session_period_val = sdt_config.get_config_value("SESSIONPERIOD") or 48.0
        session_period_sec = session_period_val * HOUR_IN_SECONDS
        
        session_times = np.zeros(2)
        session_times[1] = get_current_day_start()
        session_times[0] = session_times[1] - session_period_sec
        
        return session_times
