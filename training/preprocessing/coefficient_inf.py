from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import sys
import os

from config.sdt_constants import OFFSET, SCALE

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.data_point import DataPoint

# --- Constants ---
MAX_INDEX = 0
MIN_INDEX = 1
NETRANGE = 0.9


def get_uniform_list(stat_list: List[Optional[DataPoint]]) -> Optional[List[DataPoint]]:
    """
    Calculates a uniform set of scale and offset coefficients for an entire list of statistics.
    
    This function finds the global maximum and minimum across all provided 
    statistical data points and uses them to calculate a single [offset, scale] 
    pair that is then applied to every cycle in the session. This ensures a 
    consistent normalization across all pattern cycles.

    Args:
        stat_list (List[Optional[DataPoint]]): A list of DataPoints containing 
            per-cycle [max, min, mean, sigma].

    Returns:
        Optional[List[DataPoint]]: A list where every DataPoint has the same 
            globally-calculated scale and offset, or None if calculation fails.
    """
    if not stat_list or stat_list[0] is None:
        return None

    # Step 1: Efficiently extract all cycle extremes using NumPy
    max_values = np.array([dp.data[MAX_INDEX] for dp in stat_list if dp])
    min_values = np.array([dp.data[MIN_INDEX] for dp in stat_list if dp])

    if max_values.size == 0:
        return None
        
    # Step 2: Identify the global range of the session
    _mAx = np.max(max_values)
    _mIn = np.min(min_values)

    # Step 3: Calculate the linear transformation coefficients
    if _mAx > _mIn:
        diff = _mAx - _mIn
        # Transformation formula: scale * (raw - offset)
        # offset is the center point, scale maps the full range to [-NETRANGE, +NETRANGE]
        scale_offset = np.array([0.5 * (_mAx + _mIn), 2.0 * NETRANGE / diff], dtype=np.float32)

        return [DataPoint(dp.time, scale_offset) for dp in stat_list if dp]

    return None


class CoefficientInf(ABC):
    """
    Abstract Base Class (ABC) defining the interface for normalization coefficient calculation.
    
    Implementations of this interface provide different strategies for 
    determining the scale and offset parameters required to normalize telemetry 
    data before it is passed to a machine learning model.
    """

    @abstractmethod
    def get_coefficient(self, stat_list: List[Optional[DataPoint]]) -> Optional[List[DataPoint]]:
        """
        Calculates the normalization coefficients based on per-cycle statistics.

        Args:
            stat_list (List[Optional[DataPoint]]): The list of statistics for 
                each pattern cycle in the session.

        Returns:
            Optional[List[DataPoint]]: A list of time-tagged coefficient 
                DataPoints [offset, scale], or None.
        """
        pass
