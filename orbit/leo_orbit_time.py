import logging
import os
import sys
from typing import List, Optional

import numpy as np

import plugin_manager
from config.sdt_constants import DEFAULT
from training import data_buffer
from training.training_set import TrainingSet
from util import sdt_util, time_util

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import sdt_config
from algorithm.algorithm_def import AlgorithmDef
from orbit.model_time import ModelTime
from orbit.orbit_if import OrbitIF
from util.time_util import get_time_tag_from_seconds, get_current_day_start
from orbit.pattern_time_evaluate import PatternTimeEvaluate

# --- Constants ---
DAY_IN_SECONDS = 86400
HOUR_IN_SECONDS = 3600
CONTEXT = "LEOORBITTIME"


def _calculate_mean_period(crossing_lists: List[np.ndarray]) -> float:
    """
    Calculates the average orbital period based on multiple lists of zero-crossings.

    Args:
        crossing_lists (List[np.ndarray]): A list of arrays, where each array contains 
            timestamps of consecutive ascending node crossings.

    Returns:
        float: The mean orbital period in seconds, or 0.0 if no crossings are found.
    """
    periods = []
    for crossings in crossing_lists:
        if len(crossings) > 1:
            periods.extend(np.diff(crossings))

    return float(np.mean(periods)) if periods else 0.0


def _generate_pattern_times(crossing_lists: List[np.ndarray], mean_period: float, session_end: float) -> np.ndarray:
    """
    Constructs a continuous sequence of pattern cycle start times across a training session.
    
    It uses detected zero-crossings and fills in gaps or extrapolates at the ends of 
    the session using the calculated mean period.

    Args:
        crossing_lists (List[np.ndarray]): Lists of detected crossing timestamps.
        mean_period (float): The mean orbital period in seconds.
        session_end (float): The end timestamp of the current session.

    Returns:
        np.ndarray: A sorted array of timestamps defining each pattern cycle.
    """
    pattern_times : List[float] = []
    if not crossing_lists:
        return np.array([])
        
    pattern_times.extend(crossing_lists[0].tolist())
    
    for index in range(1, len(crossing_lists)):
        # Calculate how many missing periods fit in the gap
        gap_duration = crossing_lists[index][0] - pattern_times[-1]
        num_missing = round(gap_duration / mean_period)
        
        if num_missing > 1:
            for j in range(1, num_missing):
                pattern_times.append(pattern_times[-1] + mean_period)
        pattern_times.extend(crossing_lists[index].tolist())

    # Extrapolate to the end of the session
    last_time = pattern_times[-1]
    while session_end - last_time > mean_period:
        last_time += mean_period
        pattern_times.append(last_time)

    # Prepend cycles to the start of the session
    first_time = pattern_times[0]
    while first_time - data_buffer.session_start > mean_period:
        first_time -= mean_period
        pattern_times.insert(0, first_time)

    return np.sort(np.array(pattern_times))

def _find_crossings(data_segment: TrainingSet) -> np.ndarray:
    """
    Identifies ascending node zero-crossings (negative to positive) in positional data.
    
    Uses vectorized linear interpolation between data points to find the 
    precise timestamp when the value crosses zero.

    Args:
        data_segment (TrainingSet): A continuous segment of satellite position data.

    Returns:
        np.ndarray: An array of precise crossing timestamps.
    """
    times = data_segment.inputs[:, 0]
    values = data_segment.raw

    # Find indices where values cross from negative to non-negative
    is_crossing = (values[:-1] < 0.0) & (values[1:] >= 0.0)
    indices = np.where(is_crossing)[0]

    if indices.size == 0:
        return np.array([])

    v_prev = values[indices]
    v_curr = values[indices + 1]
    t_prev = times[indices]
    t_curr = times[indices + 1]

    # Precise crossing time using linear interpolation
    crossing_times = t_prev - v_prev * (t_curr - t_prev) / (v_curr - v_prev)

    return crossing_times


class LEOOrbitTime(OrbitIF):
    """
    OrbitIF implementation for Low Earth Orbit (LEO) satellites.
    
    This class specializes in modeling time contexts for satellites with short 
    orbital periods (e.g., ~100 minutes). it calculates pattern cycles by 
    detecting ascending node crossings in positional telemetry.
    """

    def __init__(self):
        """Initializes the LEO orbit time model components."""
        self.evaluate = PatternTimeEvaluate()
        self.sample_time = sdt_config.get_float_property("SAMPLETIME") or 25.0

    def get_default_model_time_for_prev_session(self, mnemonic_id: str) -> ModelTime:
        """
        Retrieves the time context for the previous session. 
        For LEO, this is the same as the current default calculation.

        Args:
            mnemonic_id (str): The mnemonic used for orbital context.

        Returns:
            ModelTime: The calculated time context.
        """
        return self.get_default_model_time(mnemonic_id)

    def get_default_model_time(self, mnemonic_id: str) -> ModelTime:
        """
        Calculates the pattern cycles and reference times for a LEO satellite.
        
        It retrieves orbital position telemetry, segments it to handle data gaps, 
        detects zero-crossings, and calculates the mean orbital period to generate 
        a consistent set of pattern start times.

        Args:
            mnemonic_id (str): The specific mnemonic (e.g., NPP/ADC/ADAEPOSZ) 
                used as the orbital position indicator.

        Returns:
            ModelTime: A fully populated ModelTime object for the session.
        """
        session_start = data_buffer.session_start
        session_end = data_buffer.session_end
        sat_id = sdt_config.sat_id

        # Use configured orbit mnemonic if not provided
        orbit_mnemonic = sdt_config.get_config_value("ORBITMNEMONIC") or mnemonic_id
        if orbit_mnemonic is None:
            logging.error(f"{CONTEXT}: ORBITMNEMONIC not configured for {sat_id}")
            return ModelTime(session_start, session_end, "unknown")

        model_time = ModelTime(session_start, session_end, orbit_mnemonic)
        data_input = plugin_manager.get_sdt_data_input(DEFAULT)
        training_set = data_input.get_data(mnemonic_id=orbit_mnemonic, start=session_start, end=session_end)

        if training_set is None:
            logging.warning(f"{CONTEXT}: No satellite position data found for {orbit_mnemonic}.")
            return model_time

        sdt_util.sort_training_set(training_set)
        data_segments = self._check_data_gap(training_set)

        crossing_lists = []
        for segment in data_segments:
            crossings = _find_crossings(segment)
            if crossings.size > 0:
                crossing_lists.append(crossings)

        if not crossing_lists:
            logging.warning(f"{CONTEXT}: No ascending node crossings detected.")
            return model_time

        mean_period = _calculate_mean_period(crossing_lists)
        if mean_period == 0:
            logging.warning(f"{CONTEXT}: Could not determine mean orbital period.")
            return model_time

        pattern_times = _generate_pattern_times(crossing_lists, mean_period, session_end)
        ref_time = pattern_times[0]
        
        model_time.set_pattern_times(pattern_times)
        model_time.set_reference_time(ref_time)
        model_time.set_model_period(mean_period)
        
        logging.info(f"{CONTEXT}: Reference Time: {get_time_tag_from_seconds(ref_time)}")
        logging.info(f"{CONTEXT}: Pattern Period: {mean_period / 60:.2f} minutes")
        return model_time

    def _check_data_gap(self, training_set: TrainingSet) -> List[TrainingSet]:
        """
        Identifies gaps in the telemetry stream and segments the data into 
        continuous blocks.

        Args:
            training_set (TrainingSet): The full session telemetry.

        Returns:
            List[TrainingSet]: A list of continuous data segments.
        """
        gap_threshold = 4.5 * self.sample_time
        segments : List[TrainingSet] = []
        
        input_times = training_set.inputs[:, 0]
        if len(input_times) == 0:
            return []
            
        start_segment_time = input_times[0] - 1
        for i in range(1, len(input_times)):
            delta_time = input_times[i] - input_times[i-1]
            if delta_time > gap_threshold:
                sub_set = sdt_util.get_subset_by_time(training_set, start_segment_time, input_times[i-1])
                segments.append(sub_set)
                start_segment_time = input_times[i]

        # Add the final segment
        sub_set = sdt_util.get_subset_by_time(training_set, start_segment_time, input_times[-1] + 1)
        segments.append(sub_set)
        
        return segments

    def get_model_time(self, alg: AlgorithmDef, mnemonic_id: str, training_set: TrainingSet) -> ModelTime:
        """
        Refines the model time context using the PatternTimeEvaluate tool.

        Args:
            alg (AlgorithmDef): The algorithm definition.
            mnemonic_id (str): Mnemonic ID.
            training_set (TrainingSet): Data points.

        Returns:
            ModelTime: The evaluated time context.
        """
        return self.evaluate.get_model_time(alg, mnemonic_id, training_set)

    def get_input_trend_times(self, pattern_period: float, session_time: float, num_pattern_in_training: int, is_orbitbased: bool) -> np.ndarray:
        """
        Determines the time range for historical trend retrieval based on orbital cycles.

        Args:
            pattern_period (float): The orbital period.
            session_time (float): Current session time.
            num_pattern_in_training (int): Cycles to look back.
            is_orbitbased (bool): If True, use orbital period for window size.

        Returns:
            np.ndarray: Array [start_time, window_size].
        """
        input_times = np.zeros(2)
        session_period = sdt_config.get_float_property("SESSIONPERIOD") * HOUR_IN_SECONDS

        if is_orbitbased:
            input_times[0] = session_time - 2.0 * num_pattern_in_training * pattern_period
            input_times[1] = num_pattern_in_training * pattern_period
        else:
            input_times[0] = session_time - session_period / 2
            input_times[1] = session_period / 4
            
        return input_times

    def get_session_time(self) -> np.ndarray:
        """
        Provides default session boundaries for LEO missions.

        Returns:
            np.ndarray: Array [start, end] covering the last 2 days.
        """
        session_times = np.zeros(2)
        session_times[1] = get_current_day_start()
        session_times[0] = session_times[1] - 2 * DAY_IN_SECONDS
        return session_times
