import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import math
import numpy as np

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import sdt_config
from sdtdb import sdt_db
from algorithm.algorithm_def import AlgorithmDef
from orbit.model_time import ModelTime
from orbit.orbit_if import OrbitIF

from training.training_set import TrainingSet
from util.time_util import get_time_from_string, get_time_tag_from_seconds, get_day_start

# Constants
DAY_IN_SECONDS = 86400
EARTHYEAR_IN_SECONDS = 31557600
INFO = "INFO"
CONTEXT = "LTORBITTIME"
SHIFTTYPE = "shifttype"
SHIFTPERIOD = "shiftperiod"
MAX_INDEX = 0
MIN_INDEX = 1

# Placeholder for PolyValidator

class LongTermTrendTime(OrbitIF):
    """
    Defines the time model for long-term data trends.
    """

    def __init__(self):
        self.sample_time = sdt_config.get_config_value("SAMPLETIME") or 1.0
        m_start_string = sdt_config.get_config_value("MISSIONSTART")
        self.mission_start = get_time_from_string(m_start_string) if m_start_string else 0.0

    def get_default_model_time_for_prev_session(self, mnemonics_id: str) -> Optional[ModelTime]:
        return None

    def get_default_model_time(self, mnemonics_id: str) -> Optional[ModelTime]:
        """
        Creates a default ModelTime for a long-term trend based on the overall session start and end times.
        """
        # Delayed import to avoid circular dependency
        from training.data_buffer import session_start, session_end
        
        # Check if path exists in DB (placeholder logic, assuming sdt_db has this check or we proceed)
        # In Python sdt_db, we might check if get_mnemonic_type returns something
        if  sdt_db.get_mnemonic_type(mnemonics_id):
            model_time = ModelTime(session_start, session_end, mnemonics_id)
            
            duration = session_end - session_start
            num_patterns = int(math.ceil(duration / EARTHYEAR_IN_SECONDS))
            if num_patterns == 0:
                num_patterns = 1
                
            crossing_time = session_start + np.arange(num_patterns) * EARTHYEAR_IN_SECONDS
            
            model_time.set_pattern_times(crossing_time)
            model_time.set_reference_time(session_start)
            model_time.set_model_period(EARTHYEAR_IN_SECONDS)
            return model_time
        else:
            return None

    def get_model_time(self, alg: AlgorithmDef, mnemonics_id: str, training_set: TrainingSet) -> ModelTime:
        """
        Generates a ModelTime based on the algorithm definition and provided data.
        """
        # Delayed import
        from training.data_buffer import get_default_model_time, session_start, session_end
        
        default_model_time = get_default_model_time(mnemonics_id)
        shifttype = alg.get_attribute(SHIFTTYPE)
        shiftperiod = alg.get_attribute(SHIFTPERIOD)
        
        period = None
        if shiftperiod:
            period = self._get_shift_period(shiftperiod)
            
        if not shifttype:
            return default_model_time
        else:
            # Assuming data_list is sortable or already sorted. 
            # If TimeData is a wrapper, we assume it handles access.
            # Here we need to convert TimeData to TrainingSet for analysis
            
            model_time = ModelTime(session_start, session_end, mnemonics_id)
            pattern_period = default_model_time.get_model_period()
            model_time.set_model_period(pattern_period)

            
            pattern_time = 0.0
            if shifttype in ["min", "max"]:
                minmax_times = self._get_min_max_time(training_set, period)
                if shifttype == "min":
                    pattern_time = minmax_times[MIN_INDEX]
                else:
                    pattern_time = minmax_times[MAX_INDEX]
            elif shifttype == "sdr":
                pattern_time = self._get_sdr_pattern_time(training_set)
                
            if pattern_time > 0:
                # Get start/end from data_list
                # Assuming TimeData has get_data_point and get_sample_size
                times = training_set.inputs[:, 0]
                if times:
                    _start = times[0]
                    _end = times[-1]
                    
                    num_before = int(math.ceil((pattern_time - _start) / pattern_period))
                    num_after = int(math.ceil((_end - pattern_time) / pattern_period))
                    
                    start_pattern_time = pattern_time - num_before * pattern_period
                    total_patterns = num_before + num_after + 1
                    pattern_times = start_pattern_time + np.arange(total_patterns) * pattern_period
                    
                    logging.info(f"{CONTEXT}: LT ORBIT TIME: {[get_time_tag_from_seconds(t) for t in pattern_times]}")
                    
                    model_time.set_pattern_times(pattern_times)
                    model_time.set_reference_time(pattern_times[0])
                    return model_time
            
            return default_model_time

    def get_input_trend_times(self, pattern_period: float, session_time: float, num_pattern_in_training: int, is_orbitbased: bool) -> np.ndarray:
        """
        Calculates the start and end times for the input data required for training.
        """
        input_times = np.zeros(2)
        mission_start_val = sdt_config.get_config_value("MISSIONSTART")
        # Convert string to time if needed, or assume it's already float if config handles it
        # Based on __init__, we parsed it. But here we fetch again.
        # Let's use self.mission_start
        
        input_times[0] = self.mission_start - DAY_IN_SECONDS
        input_times[1] = input_times[0] + pattern_period
        
        if is_orbitbased:
            input_times[1] = session_time - num_pattern_in_training * pattern_period
            input_times[0] = input_times[1] - num_pattern_in_training * pattern_period
            
            if input_times[0] < (self.mission_start - pattern_period):
                input_times[0] = self.mission_start - pattern_period
                input_times[1] = input_times[0] + num_pattern_in_training * pattern_period
                
        return input_times

    def get_session_time(self) -> np.ndarray:
        """
        Gets the overall time range for the long-term trend.
        """
        lt_period = np.zeros(2)
        lt_period[0] = self.mission_start
        current = datetime.now(timezone.utc).timestamp()
        lt_period[1] = get_day_start(current)
        return lt_period

    def _get_shift_period(self, shiftperiod: str) -> np.ndarray:
        tokens = shiftperiod.split('|')
        period = np.zeros(2)
        for i in range(min(2, len(tokens))):
            period[i] = get_time_from_string(tokens[i])
        return period

    def _get_min_max_time(self, training_set: TrainingSet, shift_period: Optional[np.ndarray]) -> np.ndarray:
        minmax_times = np.zeros(3)
        min_val = np.inf
        max_val = -np.inf
        
        outputs = training_set.raw
        inputs = training_set.inputs
        dqf = training_set.dqf
        
        # Vectorized approach
        times = inputs[:, 0]
        
        # Create mask for valid points (dqf == 1)
        valid_mask = (dqf == 1)
        
        # Create mask for time period if specified
        if shift_period is not None:
            time_mask = (times > shift_period[0]) & (times < shift_period[1])
            valid_mask = valid_mask & time_mask
            
        if not np.any(valid_mask):
            return minmax_times
            
        valid_outputs = outputs[valid_mask]
        valid_times = times[valid_mask]
        
        # Find min
        min_idx = np.argmin(valid_outputs)
        min_val = valid_outputs[min_idx]
        minmax_times[MIN_INDEX] = valid_times[min_idx]
        
        # Find max
        max_idx = np.argmax(valid_outputs)
        max_val = valid_outputs[max_idx]
        minmax_times[MAX_INDEX] = valid_times[max_idx]
        
        return minmax_times

    def _get_sdr_pattern_time(self, training_set: TrainingSet) -> float:
        """
        Calculates the pattern reference time based on the point of maximum second derivative.
        """
        outputs = training_set.outputs
        inputs = training_set.inputs
        dqf = training_set.dqf
        
        if len(outputs) < 3:
            return 0.0
        
        max_sdr = 0.0
        sdr_time = 0.0
        
        before = outputs[0]
        middle = outputs[1]
        time_b = inputs[0, 0]
        time_m = inputs[1, 0]
        
        for i in range(1, len(outputs) - 1):
            if dqf[i+1] == 1:
                after = outputs[i+1]
                time_a = inputs[i+1, 0]
                
                # Avoid division by zero
                if (time_m - time_b) == 0 or (time_a - time_m) == 0 or (time_a - time_b) == 0:
                     # Update state and continue
                    before = middle
                    middle = after
                    time_b = time_m
                    time_m = time_a
                    continue

                first_order_der0 = (middle - before) / (time_m - time_b)
                first_order_der1 = (after - middle) / (time_a - time_m)
                
                second_order_derivative = abs((first_order_der1 - first_order_der0) * 2.0 / (time_a - time_b))
                
                if second_order_derivative > max_sdr:
                    max_sdr = second_order_derivative
                    sdr_time = time_m
                
                before = middle
                middle = after
                time_b = time_m
                time_m = time_a
                
        return sdr_time
