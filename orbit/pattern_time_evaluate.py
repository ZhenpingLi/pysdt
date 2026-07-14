import gc
import logging
import os
import sys

import numpy as np

import training.data_buffer as db
from config.sdt_constants import DEFAULT
from sdtdb import sdt_db
from training.training_set import TrainingSet
from util import time_util

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithm.algorithm_def import AlgorithmDef
from orbit.model_time import ModelTime
from training import data_buffer


def _check_data_gaps(times: np.ndarray, pattern_period: float, frequency: float) -> float:
    """
    Analyzes the input timestamps for significant gaps relative to the expected frequency.

    Args:
        times (np.ndarray): Array of timestamps to analyze.
        pattern_period (float): The expected duration of a single pattern cycle.
        frequency (float): The expected data sampling period in seconds.

    Returns:
        float: The ratio of total gap time to the pattern period (0.0 to 1.0).
    """
    if len(times) < 2:
        return 1.0

    if frequency <= 0:
        frequency = 4.0
    gap_threshold = frequency * 3.0
    diffs = np.diff(times)
    gaps = diffs[diffs > gap_threshold]
    if gaps.size == 0:
        return 0.0
    else:
        return float(gaps.sum() / pattern_period)


class PatternTimeEvaluate:
    """
    Refines pattern cycle boundaries by evaluating features in the telemetry data.
    
    This class supports dynamic re-alignment of training windows based on 
    data events like peak values ('max'), minimums ('min'), or large jumps ('diff'). 
    It ensures that each training cycle starts at a consistent physical phase of 
    the satellite's operation.
    """
    CONTEXT = "PATTERNTIMEEVALUATE"
    ULIMIT = 1.06
    LLIMIT = 0.94

    def __init__(self):
        """Initializes the PatternTimeEvaluate instance."""
        self.sample_time: float = 1.0
        self.shift_num: float = 5.0

    def get_model_time(self, alg: AlgorithmDef, frequency: float, training_set: TrainingSet) -> ModelTime:
        """
        Coordinates the refinement process for a mnemonic's temporal context.
        
        It first retrieves the default (nominal) model time and then triggers 
        the data-driven refinement if valid parameters are available.

        Args:
            alg (AlgorithmDef): The algorithm definition.
            frequency (float): The data sampling frequency.
            training_set (TrainingSet): The session data used for evaluation.

        Returns:
            ModelTime: The refined ModelTime object.
        """
        default_model_time = data_buffer.get_default_model_time(alg)
        pattern_period = default_model_time.get_model_period()
        orig_pattern_times = default_model_time.get_pattern_times()

        if pattern_period <= 0 or orig_pattern_times is None or orig_pattern_times.size == 0:
            return default_model_time

        self.sample_time = frequency
        input_times = training_set.inputs[:, 0]
        if input_times.size == 0:
             return default_model_time
             
        first_data_time = input_times[0]
        p_0 = int((orig_pattern_times[0] - first_data_time) / pattern_period)
        
        if p_0 > 0:
            prepended_times = orig_pattern_times[0] - np.arange(p_0, 0, -1) * pattern_period
            p_times = np.concatenate([prepended_times, orig_pattern_times])
        else:
            p_times = orig_pattern_times
            
        return self._calculate_model_time(alg, pattern_period, p_times, training_set)

    def _calculate_model_time(self, alg: AlgorithmDef, pattern_period: float, o_pattern_times: np.ndarray, training_set: TrainingSet) -> ModelTime:
        """
        Performs the data-driven detection of pattern boundaries.
        
        Iterates through the nominal cycles and looks for specific data 
        features (min, max, or max diff) to determine the actual start 
        of the physical pattern.

        Args:
            alg (AlgorithmDef): The algorithm definition.
            pattern_period (float): The cycle duration in seconds.
            o_pattern_times (np.ndarray): The initial nominal pattern start times.
            training_set (TrainingSet): Data points and values.

        Returns:
            ModelTime: A ModelTime object containing the refined pattern start times.
        """
        start_time = min(db.session_start, o_pattern_times[0])
        end_time = db.session_end
        
        model_time_id = alg.get_attribute("modeltime") or DEFAULT
        shift_type = sdt_db.get_algorithm_attribute(model_time_id, "shifttype")
        
        model_time = ModelTime(start_time, end_time, model_time_id)
        model_time.set_model_period(pattern_period)
        
        shift_num_string = sdt_db.get_algorithm_attribute(model_time_id, "shiftnum")
        self.shift_num = float(shift_num_string) if shift_num_string else 5.0
        
        p_times = []
        start_index = 0
        input_times = training_set.inputs[:, 0]
        values = training_set.raw
        
        for i in range(len(o_pattern_times) - 1):
            # Extract data for the current cycle
            # Use searchsorted for efficiency if input_times is large
            mask = (input_times >= o_pattern_times[i]) & (input_times < o_pattern_times[i+1])
            times_array = input_times[mask]
            value_array = values[mask]
            
            gaps = _check_data_gaps(times_array, pattern_period, frequency=self.sample_time)
            
            if gaps < 0.1 and len(value_array) >= 2:
                if shift_type in ("max", "lmax"):
                    max_index = np.argmax(value_array)
                    p_times.append(times_array[max_index])
                elif shift_type in ("min", "lmin"):
                    min_index = np.argmin(value_array)
                    p_times.append(times_array[min_index])
                elif shift_type == "diff":
                    diffs = np.abs(np.diff(value_array))
                    if diffs.size > 0:
                        max_index = np.argmax(diffs)
                        p_time = (times_array[max_index] + times_array[max_index+1]) / 2.0
                        p_times.append(p_time)
                    elif i > 0:
                        p_times.append(p_times[i-1] + pattern_period)
            else:
                if gaps >= 0.1:
                    logging.warning(f"[{self.CONTEXT}] Large data gap ({gaps:.2f}) in {model_time_id} for cycle {i}")
                if i > 0:
                    p_times.append(p_times[i-1] + pattern_period)
        
        if not p_times:
             p_times = o_pattern_times.tolist()

        if (p_times[0] - start_time) > pattern_period:
            p_times.insert(0, p_times[0] - pattern_period)

        for pt in p_times:
            logging.info(f"[{self.CONTEXT}] Refined PATTERNTIME: {time_util.get_time_tag_from_seconds(pt)}")
            
        model_time.set_pattern_times(np.array(p_times))
        model_time.set_reference_time(p_times[0])
        gc.collect()
        return model_time
