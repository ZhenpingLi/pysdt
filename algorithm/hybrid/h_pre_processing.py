import os
import sys
from typing import List, Optional

import numpy as np

from algorithm.fbnn.nn_algorithm_factory import FBNN
from algorithm.hybrid.hybrid_trend import HybridTrend
from algorithm.stnet.state_nn_trend import StateNNTrend
from config.sdt_constants import DEFAULT
from training import data_buffer
from training.preprocessing.ex_zone import ExZone

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.data_point import DataPoint
from training.training_set import TrainingSet
from training.preprocessing.data_stat import DataStat
from training.preprocessing.orbit_based_transform import OrbitBasedTransform
from util import sdt_util

# --- Constants ---
CONTEXT = "HPreProcessing"


def get_training_sets(training_set: TrainingSet, h_trend: HybridTrend) -> List[Optional[TrainingSet]]:
    """
    Segments a session-wide TrainingSet into multiple state-specific datasets.
    
    It uses the HybridTrend's internal state index to identify the active 
    operational state for every data point and creates a new TrainingSet 
    for each identified segment. It also triggers per-state statistical 
    calculation and orbital normalization.

    Args:
        training_set (TrainingSet): The complete session telemetry.
        h_trend (HybridTrend): The composite model containing state definitions.

    Returns:
        List[Optional[TrainingSet]]: A list of datasets, one for each state 
            index in the hybrid model.
    """
    trends = h_trend.get_data_trends()
    num_states = len(trends)
    alg = h_trend.get_algorithm()
        
    inputs = training_set.inputs
    outputs = training_set.outputs
    dqf = training_set.dqf
        
    # Step 1: Bulk identification of state for every timestamp
    times = inputs[:, 0]
    state_indices = np.array([h_trend.get_state_index(t) for t in times], dtype=int)
        
    training_sets: List[Optional[TrainingSet]] = [None] * num_states
        
    # Step 2: Slice and group data by state index
    for state_idx in range(num_states):
        mask = (state_indices == state_idx)
        if not np.any(mask):
            continue
                
        state_inputs = inputs[mask]
        state_outputs = outputs[mask]
            
        state_set = TrainingSet(mnemonic_id=training_set.mnemonic_id,
                                inputs=state_inputs,
                                raw=state_outputs,
                                dqf=dqf[mask],
                                state=trends[state_idx].get_state(),
                                outputs=state_outputs)
        training_sets[state_idx] = state_set

    # Step 3: Handle dynamic temporal shifts for default states
    shift = alg.get_attribute("shifttype")
    if shift:
        for s, trend in enumerate(trends):
            if trend.get_state() == DEFAULT:
                model_times = data_buffer.get_session_model_time(alg)
                if model_times:
                    trend.set_data_model_time(model_times.get_pattern_times(), model_times.get_model_period())
        
    # Step 4: Calculate per-state statistics and apply transformations
    get_stats(training_sets, h_trend)
    perform_orbit_transform(training_sets, h_trend)
        
    return training_sets


def get_stats(training_sets: List[Optional[TrainingSet]], h_trend: HybridTrend) -> None:
    """
    Calculates statistical baselines for every state-specific training set.
    
    It maps telemetry points to their respective orbital pattern cycles and 
    computes [max, min, mean, sigma]. It attempts to reuse baseline statistics 
    from previous sessions where applicable.

    Args:
        training_sets (List[Optional[TrainingSet]]): The segmented datasets.
        h_trend (HybridTrend): The composite model.
    """
    data_stat = DataStat()
    trends = h_trend.get_data_trends()
        
    for i, training_set in enumerate(training_sets):
        if training_set:
            state = training_set.state
            pattern_times = trends[i].get_pattern_times()
            if pattern_times is None:
                continue
                
            num_patterns = len(pattern_times) - 1 if state == DEFAULT else len(pattern_times) // 2
            stat_list: List[Optional[DataPoint]] = [None] * num_patterns
            input_list = trends[i].get_stat_list()
                
            for j in range(num_patterns):
                p_index = j if state == DEFAULT else 2 * j
                
                # Try to inherit stats from the baseline model first
                if input_list:
                    input_stat = _get_input_stat(input_list, pattern_times[p_index])
                    if input_stat:
                        stat_list[j] = input_stat
                            
                # Recalculate if no baseline exists
                if stat_list[j] is None:
                    sub_set = sdt_util.get_subset_by_time(training_set, pattern_times[p_index], pattern_times[p_index + 1])
                    if sub_set:
                        stat_list[j] = data_stat.get_stat(sub_set, pattern_times[p_index], None, False)
                
            training_set.stat_list = stat_list
            trends[i].set_stats(stat_list)

def _get_input_stat(input_list: List[DataPoint], time: float) -> Optional[DataPoint]:
    """
    Searches a list of statistics for a record matching the specified timestamp.
    """
    for input_stat in input_list:
        if input_stat and abs(input_stat.time - time) < 20:
            return input_stat
    return None

def perform_orbit_transform(sets: List[Optional[TrainingSet]], h_trend: HybridTrend):
    """
    Executes orbital normalization (OrbitBasedTransform) on state components 
    that require it (e.g., Neural Network states).

    Args:
        sets (List[Optional[TrainingSet]]): Segmented datasets.
        h_trend (HybridTrend): The composite model.
    """
    trends = h_trend.get_data_trends()
    alg = h_trend.get_algorithm()
    socoef = alg.get_so_coef()
        
    for i, trend in enumerate(trends):
        if isinstance(trend, StateNNTrend) or trend.get_alg_name() == FBNN:
            transform = OrbitBasedTransform(alg, trend.mnemonic_id)
            if socoef is not None:
                transform.set_so_coef(np.array(socoef, dtype=np.float32))
                
            if sets[i]:
                # Perform the scale/offset transformation
                transform.transform(sets[i], 0)
                trend.set_stats(sets[i].stat_list)
                trend.set_scale_offset_list(sets[i].coef_list)
        else:
            if sets[i]:
                # Non-NN states typically don't require normalization
                sets[i].outputs = sets[i].raw

def update_state_zone(training_set: TrainingSet, zone_diff: float, h_trend: HybridTrend) -> bool:
    """
    Dynamically adjusts operational state boundaries by detecting jumps in 
    the telemetry signal.
    
    This handles cases where predicted state changes (e.g., maneuver start) 
    do not perfectly align with the telemetry features.

    Args:
        training_set (TrainingSet): Raw telemetry.
        zone_diff (float): The minimum jump required to trigger an update.
        h_trend (HybridTrend): The composite model with current zones.

    Returns:
        bool: True if any state boundaries were modified, False otherwise.
    """
    outputs_arr = training_set.outputs
    inputs_arr = training_set.inputs
    state_zones = h_trend.state_zones
    
    # 1. Vectorized detection of signal jumps
    diffs = np.diff(outputs_arr)
    valid_indices = np.where(diffs > zone_diff)[0]
    valid_times = inputs_arr[valid_indices, 0]
    
    updated = False
    for zones in (z.get_zones() for z in state_zones if z):
        for time in valid_times:
            # Shift zone end time to the point of the detected jump
            for zone in zones:
                if zone[0] <= time < zone[1] and (zone[1] - time) < 1800:
                    zone[1] = time
                    updated = True
                    break
    return updated

def get_ex_zones(ex_zone: ExZone, start: float, end: float, pad_factor: Optional[List[float]]) -> Optional[List[List[float]]]:
    """
    Retrieves and optionally pads exclusion zone segments within a time window.
    """
    if ex_zone:
        if pad_factor:
            zones = ex_zone.get_zones()
            padded_zones = [[zone[0] - pad_factor[0], zone[1] + pad_factor[1]] for zone in zones]
            return [z for z in padded_zones if end > z[0] >= start or end > z[1] >= start]
        else:
            return [z.tolist() for z in ex_zone.get_zones() if end > z[0] >= start or end > z[1] >= start]
    return None
