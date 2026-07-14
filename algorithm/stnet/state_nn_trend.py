import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from config import sdt_config
from config.sdt_constants import SCALE, OFFSET, MDUMP, LONGTERM, SIGMA_INDEX, MEAN_INDEX
from sdtdb.sdt_db import StateType

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.data_trend import DataTrend
from algorithm.fbnn.neural_net_wrapper import NeuralNet

# --- Constants ---
CONTEXT = "StateNNTrend"


class StateNNTrend(DataTrend):
    """
    Predictive model implementation for State-based Neural Networks (STNET).
    
    This class specializes the Neural Network model for specific operational 
    states (e.g., maneuvers, dumps). It manages a dedicated NeuralNet instance 
    and handles state-specific time normalization and coefficient scaling. 
    It supports multiple pattern cycles within a single session.
    """

    def __init__(self, mnemonic_id: str):
        """
        Initializes a new StateNNTrend model.

        Args:
            mnemonic_id (str): Unique identifier for the telemetry mnemonic.
        """
        super().__init__(mnemonic_id)
        
        self.nn_struct: Optional[List[int]] = None
        self.network: Optional[NeuralNet] = None
        self.num_patterns: int = 0
        self.scales: Optional[np.ndarray] = None
        self.alg_name = "stnet"
        
        if self.algorithm:
            np_val = self.algorithm.get_np()
            if np_val > 1:
                self.num_pattern_in_training = np_val
        
        self._init_network()

    def set_state(self, state: StateType):
        """
        Overrides the model state and re-configures the network topology.
        
        The network's hidden layer sizes are retrieved from the algorithm 
        dimension array using the state's 'dim_pointer'.

        Args:
            state (StateType): The operational state configuration.
        """
        super().set_state(state)
        dim_offset = 0
        if state.dim_pointer:
            dim_offset = int(state.dim_pointer)

        if dim_offset is not None and self.algorithm:
            dimension = self.algorithm.get_dimension()
            if len(dimension) < (dim_offset + 2):
                logging.warning(f"{CONTEXT}: Dimension array length mismatch for state '{state.name}'.")
            else:
                self.nn_struct = [1, dimension[dim_offset], dimension[dim_offset+1], 1]
                self.network = NeuralNet("TANH", self.nn_struct)
                self.param_dim = self.network.get_weight_dim()

    def _init_network(self):
        """Initializes the NeuralNet using default algorithm dimensions."""
        if not self.algorithm:
            return
            
        dims = self.algorithm.get_dimension()
        self.nn_struct = [1, dims[0], dims[1], 1]
        self.network = NeuralNet("TANH", self.nn_struct)
        self.param_dim = self.network.get_weight_dim()

    def get_model_param_dim(self) -> int:
        """Calculates the total dimension required for serializing this model."""
        model_param_dim = self.param_dim + 8 * self.num_pattern_in_training
        if self.state and self.state.name == MDUMP:
            model_param_dim += 4 * self.num_pattern_in_training
        return model_param_dim

    def set_pattern_times(self, pattern_times: np.ndarray):
        """
        Configures the start/end timestamps for each pattern cycle.
        
        Also calculates the duration (scale) for each cycle to enable 
        time normalization.

        Args:
            pattern_times (np.ndarray): 1D array of [start1, end1, start2, end2...].
        """
        super().set_pattern_times(pattern_times)
        if pattern_times is not None:
            self.num_patterns = len(pattern_times) // 2
            self.scales = np.zeros(self.num_patterns)
            for i in range(self.num_patterns):
                self.scales[i] = pattern_times[2 * i + 1] - pattern_times[2 * i]
            
            if self.scales.size > 0:
                self.pattern_period = self.scales[-1]
            if pattern_times.size > 1:
                self.ref_time = pattern_times[-2]

    def get_trend_value(self, time: List[float]) -> float:
        """
        Calculates the predicted trend value for a given timestamp.
        
        Performs network inference with current weights and applies 
        inverse normalization based on the identified cycle.

        Args:
            time (List[float]): Input vector where time[0] is the timestamp.

        Returns:
            float: The predicted telemetry value.
        """
        stat = self.stat_list[0] if self.stat_list else None
        if stat is None and self.stat_list and len(self.stat_list) > 1:
            stat = self.stat_list[1]
            
        if (self.training_type == LONGTERM or (stat and stat.data[SIGMA_INDEX] > 0)) and any(self.params):
            self.network.set_weight(np.array(self.params))
            
            index = self._get_index(time[0])
            norm_time = np.array([self.get_data_model_time(time[0])])
            orig_value = self.network.get_output(norm_time)[0]
            
            # Map back to physical units: (norm_val / scale) + offset
            so_coef = self.scale_offset_list[index].data
            scale = so_coef[SCALE]
            offset = so_coef[OFFSET]
            
            return float(orig_value / scale + offset) if scale != 0 else offset
        else:
            # Fallback to cycle mean
            return float(stat.data[MEAN_INDEX]) if stat else 0.0

    def get_net_file_name(self) -> str:
        """Returns the standard file path for storing model weights."""
        sat_id = sdt_config.sat_id
        parent_dir = Path(__file__).resolve().parent.parent.parent
        state_name = self.state.name if self.state else "default"
        return os.path.join(parent_dir, "db", sat_id.lower(), "networks", 
                            f"{sat_id}-{self.mnemonic_id}-trend-{state_name}.txt")

    def get_data_model_time(self, time: float) -> float:
        """
        Normalizes an absolute timestamp to a [0, 1] range within its cycle.

        Args:
            time (float): The Unix timestamp.

        Returns:
            float: Normalized model-relative time.
        """
        index = self._get_index(time)
        if self.pattern_times is None or self.scales is None:
            return 0.0
            
        time_since = time - self.pattern_times[2 * index]
        scale = self.scales[index]
        
        return float(time_since / scale) if scale != 0 else 0.0

    def _get_index(self, time: float) -> int:
        """
        Identifies which pattern cycle contains the provided timestamp.

        Args:
            time (float): Unix timestamp.

        Returns:
            int: The index of the containing cycle.
        """
        if self.num_patterns <= 1:
            return 0
        
        if self.pattern_times is not None:
            for i in range(self.num_patterns):
                if self.pattern_times[2 * i] <= time < self.pattern_times[2 * i + 1]:
                    return i
        return 0
