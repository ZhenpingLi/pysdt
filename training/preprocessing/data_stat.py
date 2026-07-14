import os
import sys
import typing

import numpy as np

from config.sdt_constants import MAX_INDEX, MIN_INDEX, MEAN_INDEX, SIGMA_INDEX
from .ex_zone import ExZone

# Add parent directory to path to find algorithm and training
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.data_point import DataPoint
from training.training_set import TrainingSet


def cal_stat(training_set: TrainingSet, start_time: float, ezones: typing.Optional[ExZone] = None) -> DataPoint:
    """
    Calculates the descriptive statistics for a given training set.
    
    It computes the maximum, minimum, arithmetic mean, and standard deviation 
    (sigma) for all valid data points (where DQF=1 and points are not in 
    exclusion zones).

    Args:
        training_set (TrainingSet): The dataset containing telemetry values and quality flags.
        start_time (float): The reference timestamp to associate with the results.
        ezones (Optional[ExZone]): An optional exclusion zone filter.

    Returns:
        DataPoint: A DataPoint containing [max, min, mean, sigma] for the set.
    """
    output = training_set.outputs
    dqf = training_set.dqf
    inputs = training_set.inputs

    # Vectorized approach: Identify points to include
    mask = (dqf == 1)

    if ezones:
        times = inputs[:, 0]
        # Invert the zone check: mask points that are NOT in a zone
        zone_mask = np.array([not ezones.is_in_zone(t) for t in times])
        mask = mask & zone_mask

    valid_data = output[mask]

    if len(valid_data) == 0:
        stat_value = np.array([-np.inf, np.inf, 0.0, 0.0], dtype=np.float32)
        return DataPoint(time=start_time, data=stat_value)

    # Calculate statistics using optimized NumPy routines
    _min = np.min(valid_data)
    _max = np.max(valid_data)
    _mean = np.mean(valid_data)
    _sigma = np.std(valid_data)

    stat_value = np.zeros(4, dtype=np.float32)
    stat_value[MAX_INDEX] = _max
    stat_value[MIN_INDEX] = _min
    stat_value[MEAN_INDEX] = _mean
    stat_value[SIGMA_INDEX] = _sigma

    return DataPoint(time=start_time, data=stat_value)


def is_included(dqf: int, ezones: typing.Optional[ExZone], time: float) -> bool:
    """
    Helper to check if a single data point should be included in statistics.

    Args:
        dqf (int): The data quality flag (1 or 0).
        ezones (Optional[ExZone]): The exclusion zone filter.
        time (float): The timestamp of the point.

    Returns:
        bool: True if the point is valid and not in an exclusion zone.
    """
    included = (dqf == 1)
    if ezones:
        included = included and not ezones.is_in_zone(time)
    return included


class DataStat:
    """
    Statistical processor for telemetry training sets.
    
    Provides methods to calculate high-level descriptive statistics and to 
    perform iterative outlier removal by resetting Data Quality Flags (DQF) 
    based on sigma thresholds.
    """

    def __init__(self, limit: float = 6.0):
        """
        Initializes the DataStat processor.

        Args:
            limit (float): The multiplier for standard deviation (sigma) used 
                to define outlier thresholds. Defaults to 6.0.
        """
        self.limit = limit

    def reset_dqf(self, training_set: TrainingSet, stat: DataPoint, ezones: typing.Optional[ExZone] = None) -> bool:
        """
        Identifies outliers and resets their Data Quality Flags (DQF) to 0.
        
        A point is marked as an outlier if its absolute difference from the 
        mean exceeds the specified sigma threshold (limit * sigma).

        Args:
            training_set (TrainingSet): The dataset to update.
            stat (DataPoint): Pre-calculated statistics (mean and sigma).
            ezones (Optional[ExZone]): The exclusion zone filter.

        Returns:
            bool: True if any DQFs were changed, otherwise False.
        """
        output = training_set.outputs
        dqf = training_set.dqf
        inputs = training_set.inputs
        
        sigma = stat.data[SIGMA_INDEX]
        mean = stat.data[MEAN_INDEX]
        threshold = self.limit * sigma
        
        if sigma <= 0:
            return False

        # Vectorized identification of valid points
        current_valid_mask = (dqf == 1)
        if ezones:
            times = inputs[:, 0]
            zone_mask = np.array([not ezones.is_in_zone(t) for t in times])
            current_valid_mask = current_valid_mask & zone_mask
            
        # Identify outliers among the valid points
        diff = np.abs(output - mean)
        outlier_mask = (diff > threshold) & current_valid_mask
        
        if np.any(outlier_mask):
            dqf[outlier_mask] = 0
            training_set.dqf = dqf
            return True
            
        return False

    def get_stat(self, training_set: TrainingSet, time: float, ezones: typing.Optional[ExZone] = None, check_dqf: bool = True) -> DataPoint:
        """
        Orchestrates the statistical calculation, optionally performing 
        one round of outlier removal and recalculation.

        Args:
            training_set (TrainingSet): The source dataset.
            time (float): The reference timestamp.
            ezones (Optional[ExZone]): The exclusion zone filter.
            check_dqf (bool): If True, performs iterative outlier removal.

        Returns:
            DataPoint: The final calculated statistics.
        """
        stat = cal_stat(training_set, time, ezones)
        if check_dqf and self.reset_dqf(training_set, stat, ezones):
            # Recalculate if outliers were removed
            stat = cal_stat(training_set, time, ezones)
        return stat
