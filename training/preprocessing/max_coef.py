from typing import List, Optional
import numpy as np
import sys
import os

from config.sdt_constants import SCALE, OFFSET

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.data_point import DataPoint
from .coefficient_inf import CoefficientInf, get_uniform_list

# --- Constants ---
MAX_INDEX = 0
MIN_INDEX = 1
NETRANGE = 0.9


class MaxCoef(CoefficientInf):
    """
    Normalization strategy that uses the maximum value of a cycle as the offset.
    
    This implementation of CoefficientInf calculates dynamic scale and offset 
    parameters for each pattern cycle. It centers the data around the cycle's 
    maximum value and scales it to fit within a target range. If a cycle has 
    no variance (max == min), it falls back to a global uniform scaling.
    """

    def get_coefficient(self, stat_list: List[Optional[DataPoint]]) -> List[Optional[DataPoint]]:
        """
        Calculates the per-cycle normalization coefficients.
        
        The offset is set to the cycle's maximum value, and the scale is 
        derived from the cycle's range (max - min).

        Args:
            stat_list (List[Optional[DataPoint]]): A list of statistics 
                [max, min, mean, sigma] for each pattern cycle.

        Returns:
            List[Optional[DataPoint]]: A list of time-tagged DataPoints 
                containing the [offset, scale] coefficients.
        """
        scale_offset_list: List[Optional[DataPoint]] = [None] * len(stat_list)
        is_uniform = False

        for index, stat in enumerate(stat_list):
            if stat:
                diff = stat.data[MAX_INDEX] - stat.data[MIN_INDEX]
                
                if diff == 0.0:
                    is_uniform = True
                    break
                
                scale_offset = np.zeros(2, dtype=np.float32)
                # Formula: value = scale * (raw - offset)
                scale_offset[SCALE] = NETRANGE / diff
                scale_offset[OFFSET] = stat.data[MAX_INDEX]
                
                scale_offset_list[index] = DataPoint(time=stat.time, data=scale_offset)
        
        # Fallback to session-wide uniform scaling if any cycle has zero range
        if is_uniform:
            return get_uniform_list(stat_list)
        else:
            return scale_offset_list
