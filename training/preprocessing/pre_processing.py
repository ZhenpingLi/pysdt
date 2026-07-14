from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import sys
import os
import logging

from algorithm.mnemonic_node import MnemonicNode
from util import sdt_util

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import sdt_config
from algorithm.data_point import DataPoint
from algorithm.algorithm_def import AlgorithmDef
from algorithm.data_trend import DataTrend
from training.training_set import TrainingSet
from .data_stat import DataStat
from .ex_zone import ExZone


def _reset_dqf(sub_set: TrainingSet, main_set: TrainingSet):
    """
    Synchronizes Data Quality Flag (DQF) changes from a processed subset back 
    to the main training set.
    
    This is typically called after an outlier detection routine has modified 
    the DQF of a specific pattern cycle (subset).

    Args:
        sub_set (TrainingSet): The subset with potentially modified DQFs.
        main_set (TrainingSet): The original session-wide dataset.
    """
    sub_inputs = sub_set.inputs
    if sub_inputs is None or sub_inputs.shape[0] == 0:
        return

    sub_dqf = sub_set.dqf
    main_dqf = main_set.dqf
    main_inputs = main_set.inputs

    # Efficiently find the start and end indices in the main set
    start_time = sub_inputs[0, 0]
    end_time = sub_inputs[-1, 0]

    main_times = main_inputs[:, 0]
    s_index = np.searchsorted(main_times, start_time, side='left')
    e_index = np.searchsorted(main_times, end_time, side='right') - 1

    # Apply the sub-DQF mask to the identified slice
    if e_index - s_index + 1 == len(sub_dqf):
        main_dqf[s_index : e_index + 1] = sub_dqf
    else:
        # Fallback to slower dictionary-based matching if indices are inconsistent
        logging.warning("DQF synchronization using fallback method due to size mismatch.")
        sub_time_map = {time: dqf_val for time, dqf_val in zip(sub_inputs[:, 0], sub_dqf)}
        for i in range(s_index, e_index + 1):
            if main_inputs[i, 0] in sub_time_map:
                main_dqf[i] = sub_time_map[main_inputs[i, 0]]


def get_nn_training_set(original_set: TrainingSet, trend: DataTrend) -> Optional[TrainingSet]:
    """
    Transforms a standard TrainingSet into a format optimized for neural 
    network (FBNN) training.
    
    It performs the following:
    1. Filters out invalid data points (DQF=0).
    2. Normalizes timestamps to a relative scale [0, 1] within a pattern cycle.
    3. Re-sorts the data based on the new normalized time features.

    Args:
        original_set (TrainingSet): The input telemetry data.
        trend (DataTrend): The model object used to calculate model-relative time.

    Returns:
        Optional[TrainingSet]: The transformed dataset, or None if no valid 
            points remain.
    """
    inputs = original_set.inputs
    outputs = original_set.outputs
    dqf = original_set.dqf

    # Keep only points with good quality
    valid_mask = (dqf == 1)
    if not np.any(valid_mask):
        return None
        
    valid_inputs = inputs[valid_mask]
    valid_outputs = outputs[valid_mask]

    # Normalize absolute time to model-relative time (hours since cycle start)
    nn_input_time = np.array([trend.get_data_model_time(t) for t in valid_inputs[:, 0]])
    
    # Ensure inputs are sorted by the normalized time dimension
    sort_indices = np.argsort(nn_input_time)
    nn_input_time = nn_input_time[sort_indices]
    valid_outputs = valid_outputs[sort_indices]
    
    # Reconstruct input feature matrix
    if valid_inputs.shape[1] > 1:
        nn_input = np.column_stack([nn_input_time, valid_inputs[sort_indices, 1:]])
    else:
        nn_input = nn_input_time.reshape(-1, 1)

    nn_set = TrainingSet(mnemonic_id=original_set.mnemonic_id, inputs=nn_input, 
                         raw=valid_outputs, outputs=valid_outputs, 
                         dqf=np.ones(len(valid_outputs), dtype=np.int8))
                         
    nn_set.stat_list = original_set.stat_list
    nn_set.coef_list = original_set.coef_list

    return nn_set


class PreProcessing(ABC):
    """
    Abstract base class for all data preprocessing strategies.
    
    Defines the framework for preparing raw telemetry data for training workers. 
    It handles common tasks such as calculating pattern statistics, 
    managing exclusion zones, and identifying data gaps.
    """
    CONTEXT = "PREPROCESSING"
    
    def __init__(self):
        """Initializes the preprocessing component with default configurations."""
        self.gap_threshold = sdt_config.get_float_property("GAPTHRESHOLD") or 0.5
        self.ezones: Optional[ExZone] = None
        self.so_coefficients: Optional[np.ndarray] = None

    @abstractmethod
    def preprocess_training_set(self, training_set: TrainingSet, node: MnemonicNode, pattern_times: np.ndarray, pattern_period: float, pattern_offset: int) -> TrainingSet:
        """
        Main orchestration method to preprocess a dataset. Must be implemented by subclasses.

        Args:
            training_set (TrainingSet): Raw input data.
            node (MnemonicNode): Mnemonic identifier and hierarchy context.
            pattern_times (np.ndarray): Nominal start times for cycles.
            pattern_period (float): Nominal cycle duration.
            pattern_offset (int): the offset position in the pattern_times array.
        Returns:
            TrainingSet: The fully processed dataset ready for fitting.
        """
        pass
    
    def get_stats(self, training_set: TrainingSet, pattern_times: np.ndarray, sample_period: float, mnemonic_id: str, alg: AlgorithmDef, pattern_period: float):
        """
        Calculates descriptive statistics (mean, sigma) for every pattern cycle in a session.
        
        It identifies and filters outliers during the process and populates the 
        'stat_list' attribute of the training set. It also handles gap filling 
        by duplicating statistics from adjacent cycles if a gap is detected.

        Args:
            training_set (TrainingSet): Data set to process.
            pattern_times (np.ndarray): Boundaries for each pattern cycle.
            sample_period (float): Data sampling frequency.
            mnemonic_id (str): Mnemonic name.
            alg (AlgorithmDef): Algorithm configuration.
            pattern_period (float): nominal cycle duration in seconds.
        """
        limit = 18.0 if alg.get_name() == "DISCRETE" else 6.0
        data_stat = DataStat(limit)
        
        num_patterns = len(pattern_times) - 1
        stat_list: List[Optional[DataPoint]] = [None] * num_patterns
        missing_index_list = []

        for i in range(num_patterns):
            start_time, end_time = pattern_times[i], pattern_times[i+1]
            sub_set = sdt_util.get_subset_by_time(training_set, float(start_time), float(end_time))
            period = end_time - start_time

            if sub_set and len(sub_set.raw) > 0:
                gap = sdt_util.check_data_gap(sub_set, mnemonic_id, period)
                if gap < self.gap_threshold:
                    # Calculate stats and remove outliers (reset DQF)
                    stat_list[i] = data_stat.get_stat(sub_set, float(start_time), None)
                    _reset_dqf(sub_set, training_set)
                else:
                    logging.info(f"[{self.CONTEXT}] Large Data Gap ({gap:.2%}) for cycle {i}. Stats invalidated.")
                    stat_list[i] = DataPoint(time=float(start_time), data=np.zeros(4, dtype=np.float32))
                    missing_index_list.append(i)
            else:
                stat_list[i] = DataPoint(time=float(start_time), data=np.zeros(4, dtype=np.float32))
                missing_index_list.append(i)
                
        # Simple gap filling for missing statistics
        if missing_index_list:
            for m_idx in missing_index_list:
                # Find nearest valid neighbor
                valid_idx = 0
                if m_idx > 0:
                    valid_idx = m_idx - 1
                elif m_idx == 0 and num_patterns > 1:
                    valid_idx = m_idx + 1
                stat_list[m_idx].data = stat_list[valid_idx].data

        training_set.stat_list = stat_list

    def set_ex_zone(self, ezones: ExZone):
        """Registers exclusion zones to be ignored during preprocessing."""
        self.ezones = ezones
    
    def set_so_coef(self, coef: np.ndarray):
        """Sets scale and offset coefficients for normalization."""
        self.so_coefficients = coef
