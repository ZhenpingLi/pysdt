import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from config.sdt_constants import DEFAULT

# Add parent directory to path to find algorithm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithm.data_point import DataPoint

@dataclass
class TrainingSet:
    """
    Data structure representing a complete dataset for machine learning algorithms.
    
    This class serves as the primary container for data passed to training workers. 
    It maintains the separation between raw telemetry and processed/normalized 
    values used by algorithms, alongside quality indicators.

    Attributes:
        mnemonic_id (str): The unique identifier for the telemetry mnemonic.
        inputs (np.ndarray): 2D array of input features (typically [time, ...]).
        raw (np.ndarray): 1D array of original, unprocessed telemetry values.
        outputs (np.ndarray): 1D array of values used as the target for training 
            (may be normalized or transformed).
        dqf (np.ndarray): 1D integer array of Data Quality Flags (1: valid, 0: outlier).
        state (str): The operational state associated with this set. Defaults to 'DEFAULT'.
        stat_list (Optional[List[DataPoint]]): Statistical metrics calculated 
            per pattern cycle.
        coef_list (Optional[List[DataPoint]]): Model-specific normalization 
            parameters (e.g., scale/offsets).
    """
    mnemonic_id: str
    inputs: np.ndarray
    raw: np.ndarray
    outputs: np.ndarray
    dqf: np.ndarray
    state : str = DEFAULT
    stat_list: Optional[List[DataPoint]] = field(default=None)
    coef_list: Optional[List[DataPoint]] = field(default=None)
