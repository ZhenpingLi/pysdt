import os
import sys
from typing import List, Optional

import numpy as np

from algorithm.data_point import DataPoint
from training.training_set import TrainingSet

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sdtdb import sdt_db


def check_data_gap(training_set: TrainingSet, mnemonic_id: str, period: float) -> float:
    """
    Calculates the ratio of significant gaps in the telemetry data relative to a period.
    
    A gap is considered significant if the difference between consecutive 
    timestamps exceeds 6x the nominal sampling frequency of the mnemonic.

    Args:
        training_set (TrainingSet): The dataset to analyze.
        mnemonic_id (str): The identifier used to look up nominal frequency.
        period (float): The total duration of the analysis window in seconds.

    Returns:
        float: The gap ratio (0.0 to 1.0), where 1.0 indicates a complete gap 
            or missing data.
    """
    if not training_set or training_set.raw.size == 0:
        return 1.0

    mn_type = sdt_db.get_mnemonic_type(mnemonic_id)
    if not mn_type:
        return 1.0
        
    gap_threshold_value = mn_type.frequency * 6.0
    times = training_set.inputs[:, 0]

    diffs = np.diff(times)
    gaps = diffs[diffs > gap_threshold_value]

    if gaps.size == 0:
        return 0.0
    else:
        return float(gaps.sum() / period)

def sort_training_set(training_set: TrainingSet) -> None:
    """
    Ensures a TrainingSet is chronologically sorted and free of duplicate timestamps.
    
    This function modifies the training_set in-place. It first sorts all 
    associated arrays (inputs, raw, outputs, dqf) by the first column of the 
    inputs matrix (time). It then identifies and removes any rows with 
    duplicate timestamps.
    
    Args:
        training_set (TrainingSet): The dataset to sort and deduplicate.
    """
    if training_set.inputs.size == 0:
        return
        
    # Step 1: Standard temporal sorting
    sort_indices = np.argsort(training_set.inputs[:, 0])

    inputs_sorted = training_set.inputs[sort_indices]
    raw_sorted = training_set.raw[sort_indices]
    outputs_sorted = training_set.outputs[sort_indices]
    dqf_sorted = training_set.dqf[sort_indices]

    # Step 2: Identification of unique timestamps
    # np.unique returns the index of the first occurrence
    _, unique_indices = np.unique(inputs_sorted[:, 0], return_index=True)

    # Maintain original temporal order for the unique indices
    unique_indices = np.sort(unique_indices)

    # Step 3: Apply the unique mask
    training_set.inputs = inputs_sorted[unique_indices]
    training_set.raw = raw_sorted[unique_indices]
    training_set.outputs = outputs_sorted[unique_indices]
    training_set.dqf = dqf_sorted[unique_indices]

def get_subset_by_time(training_set: TrainingSet, start: float, end: float) -> Optional[TrainingSet]:
    """
    Extracts a temporal subset of a TrainingSet within the specified [start, end] window.
    
    Uses efficient binary search (np.searchsorted) to identify the index range. 
    Also correctly handles slicing of associated metadata like stat_list and coef_list.

    Args:
        training_set (TrainingSet): The source dataset.
        start (float): The start timestamp (inclusive).
        end (float): The end timestamp (exclusive).

    Returns:
        Optional[TrainingSet]: A new TrainingSet containing the subset data, 
            or None if the resulting subset would be empty.
    """
    if training_set.raw.size == 0:
        return None
        
    times = training_set.inputs[:, 0]
    # Find start and end indices using binary search
    s_index = np.searchsorted(times, start, side='left')
    e_index = np.searchsorted(times, end, side='left') - 1
    
    if e_index >= s_index:
        sub_set = get_subset_by_index(training_set, s_index, e_index)
        if sub_set:
            # Subset associated lists (stats and coefficients)
            if training_set.stat_list:
                sub_set.stat_list = _get_sub_list(start, end, training_set.stat_list)
            if training_set.coef_list:
                sub_set.coef_list = _get_sub_list(start, end, training_set.coef_list)
        return sub_set
    return None

def get_subset_by_index(training_set: TrainingSet, s_index: int, e_index: int) -> Optional[TrainingSet]:
    """
    Extracts a subset of a TrainingSet based on explicit row indices.

    Args:
        training_set (TrainingSet): Source dataset.
        s_index (int): Start index.
        e_index (int): End index (inclusive).

    Returns:
        Optional[TrainingSet]: A new TrainingSet containing copies of the 
            sliced data arrays.
    """
    size = e_index - s_index + 1
    if size <= 0:
        return None
        
    # Define the slice for Python/NumPy (exclusive end)
    sl = slice(s_index, e_index + 1)
    
    sub_inputs = training_set.inputs[sl].copy() if training_set.inputs is not None else None
    sub_raw = training_set.raw[sl].copy() if training_set.raw is not None else None
    sub_outputs = training_set.outputs[sl].copy() if training_set.outputs is not None else None
    sub_dqf = training_set.dqf[sl].copy() if training_set.dqf is not None else None

    return TrainingSet(mnemonic_id=training_set.mnemonic_id, inputs=sub_inputs, 
                       raw=sub_raw, outputs=sub_outputs, dqf=sub_dqf)

def _get_sub_list(start: float, end: float, dp_list: List[DataPoint]) -> Optional[List[DataPoint]]:
    """
    Helper to filter a list of time-tagged DataPoints by a time range.
    """
    sub_list = [dp for dp in dp_list if dp and start <= dp.time < end]
    return sub_list if sub_list else None
