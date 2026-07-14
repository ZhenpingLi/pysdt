from typing import List, Optional
import numpy as np
import sys
import os

from config.sdt_constants import OFFSET, SCALE

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.data_point import DataPoint
from .coefficient_inf import CoefficientInf

# --- Constants ---
MAX_INDEX = 0
MIN_INDEX = 1
NETRANGE = 0.9


class UniformCoef(CoefficientInf):
    """
    Normalization strategy that applies a global, uniform scaling across all cycles.
    
    This implementation of CoefficientInf identifies the global extremes (max 
    and min) across all provided pattern cycles and calculates a single set of 
    scale and offset parameters. This ensures that the entire session's data 
    is normalized to the same relative range, preserving the relative 
    magnitudes between different cycles.
    """

    def get_coefficient(self, stat_list: List[Optional[DataPoint]]) -> List[Optional[DataPoint]]:
        """
        Calculates session-wide uniform normalization coefficients.
        
        The offset is the midpoint of the global max/min, and the scale maps 
        the global spread to the target numeric range. Every cycle in the 
        returned list will have identical coefficients.

        Args:
            stat_list (List[Optional[DataPoint]]): A list of statistics 
                [max, min, mean, sigma] for each pattern cycle.

        Returns:
            List[Optional[DataPoint]]: A list of time-tagged DataPoints 
                containing the [offset, scale] coefficients.
        """
        if not stat_list:
            return []

        # Step 1: Gather valid statistics for global range calculation
        valid_stats = [dp for dp in stat_list if dp is not None]

        if not valid_stats:
            return [None] * len(stat_list)

        # Step 2: Identify the session-wide extremes using NumPy
        max_values = np.array([dp.data[MAX_INDEX] for dp in valid_stats])
        min_values = np.array([dp.data[MIN_INDEX] for dp in valid_stats])

        global_max = np.max(max_values)
        global_min = np.min(min_values)

        # Step 3: Calculate the static coefficients
        scale_offset = np.zeros(2, dtype=np.float32)
        scale_offset[OFFSET] = 0.5 * (global_max + global_min)

        diff = global_max - global_min
        if diff == 0:
            scale_offset[SCALE] = 0.0
        else:
            scale_offset[SCALE] = (2.0 * NETRANGE) / diff

        # Step 4: Map the global coefficients to each time point
        scale_offset_list = []
        for dp in stat_list:
            if dp:
                scale_offset_list.append(DataPoint(time=dp.time, data=scale_offset))
            else:
                scale_offset_list.append(None)

        return scale_offset_list
