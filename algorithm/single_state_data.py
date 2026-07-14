from dataclasses import dataclass
from typing import Optional, List

import numpy as np

from algorithm.algorithm_data import AlgorithmData
from algorithm.data_point import DataPoint
from config.sdt_constants import DEFAULT


@dataclass
class SingleStateData(AlgorithmData):
    """
    Picklable data structure representing the training results for a single operational state.
    
    This class is primarily used for serializing and transferring trained model 
    parameters and statistics between processes or for database storage. It 
    inherits from AlgorithmData.

    Attributes:
        pattern_period (float): The duration of one pattern cycle in seconds.
        pattern_times (Optional[np.ndarray]): Array of timestamps for cycle starts.
        ref_time (float): The reference start timestamp for the trend.
        stat_list (Optional[List[DataPoint]]): Statistical metrics (mean, max, min, sigma) 
            per cycle.
        scale_offset_list (Optional[List[DataPoint]]): Normalization parameters 
            (slope, offset) used by certain models.
        sigma (float): The standard deviation of the fit.
        tpc (float): The temporal change ratio compared to previous sessions.
        params (Optional[List[float]]): The trained model coefficients.
        num_pattern_in_training (int): Number of cycles used in the training window.
        state (str): The name of the operational state (e.g., 'DEFAULT', 'ECL').
    """
    pattern_period: float
    pattern_times: Optional[np.ndarray]
    ref_time: float
    stat_list: Optional[List[DataPoint]]
    scale_offset_list: Optional[List[DataPoint]]
    sigma: float
    tpc: float
    params: Optional[List[float]]
    num_pattern_in_training: int=1
    state : str = DEFAULT
