from dataclasses import dataclass
from typing import List

import numpy as np

from algorithm.algorithm_data import AlgorithmData
from algorithm.single_state_data import SingleStateData

@dataclass
class HybridStateData(AlgorithmData):
    """
    Data structure representing training results for a multi-state (Hybrid) trend model.
    
    This class is used to serialize and transfer training outputs for systems 
    modeled as finite state machines, where different operational states (e.g., 
    NORMAL vs. MANEUVER) have distinct models. It inherits from AlgorithmData.

    Attributes:
        state_zones (List[np.ndarray]): A list of time segments (start, end) 
            defining when each operational state was active.
        data_trend_list (List[SingleStateData]): A list of trained model parameters 
            and statistics for each detected state.
    """
    state_zones: List[np.ndarray]
    data_trend_list: List[SingleStateData]
