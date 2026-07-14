import os
import sys
from typing import List, Optional

import numpy as np

from config.sdt_constants import SCALE, OFFSET, FBNN, LONGTERM, SIGMA_INDEX, MEAN_INDEX, DEFAULT

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.data_trend import DataTrend, SOCOEF
from algorithm.fbnn.neural_net_wrapper import NeuralNet


class FBNNTrend(DataTrend):
    """
    Predictive model implementation for Feed-forward Back-propagation Neural Networks.
    
    This class manages a NeuralNet instance (wrapper for MLPRegressor) and 
    provides methods to perform high-precision telemetry predictions. It handles 
    dynamic time-scaling within orbital pattern cycles and performs 
    post-prediction inverse normalization to map model outputs back to physical 
    telemetry units.
    """

    def __init__(self, mnemonic_id: str):
        """
        Initializes the FBNNTrend model.

        Args:
            mnemonic_id (str): Unique identifier for the telemetry mnemonic.
        """
        super().__init__(mnemonic_id)
        self.nn_struct: Optional[List[int]] = None
        self.network: Optional[NeuralNet] = None
        self._limits: Optional[np.ndarray] = None
        self._init_network()
        self.alg_name = FBNN

    def _init_network(self):
        """
        Initializes the internal NeuralNet object based on algorithm dimensionality.
        
        The structure is typically [1 (time), hidden1, hidden2, 1 (value)].
        """
        if not self.algorithm or not self.algorithm.get_dimension():
            return
            
        dims = self.algorithm.get_dimension()
        self.nn_struct = [1, dims[0], dims[1], 1]
        
        self.network = NeuralNet("TANH", self.nn_struct)
        self.param_dim = self.network.get_weight_dim()

    def set_network(self, nn: NeuralNet):
        """
        Overrides the internal network instance.

        Args:
            nn (NeuralNet): The new neural network model.
        """
        self.network = nn
        self.nn_struct = nn.get_network_struct()

    def get_trend_value(self, time: List[float]) -> float:
        """
        Calculates the predicted trend value for a given timestamp.
        
        This involves:
        1. Normalizing the timestamp to model-relative time.
        2. Running network inference with the current weights.
        3. Applying inverse scale/offset transformation to the result.

        Args:
            time (List[float]): A vector where the first element is the timestamp.

        Returns:
            float: The predicted telemetry value.
        """
        stat = self.get_stat_at_time(time[0])
        
        if self.training_type == LONGTERM or (stat and stat.data[SIGMA_INDEX] > 0):
            if self.params is None or self.network is None:
                return 0.0

            self.network.set_weight(np.array(self.params))
            
            # Map absolute time to cycle-relative hours/seconds
            network_time = np.array([self.get_data_model_time(time[0])])
            orig_value = self.network.get_output(network_time)[0]
            
            # Perform inverse normalization
            return self.get_output(time[0], float(orig_value))
        else:
            # Fallback to mean if data is static or un-trended
            return float(stat.data[MEAN_INDEX]) if stat else 0.0

    def get_output(self, time: float, orig_value: float) -> float:
        """
        Applies inverse normalization to a raw model output.
        
        Formula: (orig_value / scale) + offset

        Args:
            time (float): The timestamp used to retrieve the relevant coefficients.
            orig_value (float): The normalized output from the neural network.

        Returns:
            float: The value in original physical units.
        """
        scale_offset = self.get_scale_offset(time)
        if scale_offset is None or len(scale_offset) < 2:
            return orig_value
            
        scale = scale_offset[SCALE]
        if scale == 0:
            return scale_offset[OFFSET]
            
        return orig_value / scale + scale_offset[OFFSET]

    def get_network(self) -> Optional[NeuralNet]:
        """Returns the underlying NeuralNet instance."""
        return self.network

    def get_weight(self) -> Optional[List[float]]:
        """Returns the current model weights."""
        return self.params

    def set_model_params(self, p: List[float], ref_time: float):
        """
        Reconstructs the full model state from a serialized parameter list.
        
        Deserializes weights, pattern periods, pattern timestamps, and 
        historical scale/offset DataPoints.

        Args:
            p (List[float]): The flat parameter array from the archive.
            ref_time (float): The reference start timestamp.
        """
        super().set_model_params(p, ref_time)
        
        if self.network and self.params:
            self.network.set_weight(np.array(self.params))

        offset = self.param_dim + 1
        if offset >= len(p):
            return

        self.pattern_period = p[offset]
        offset += 1
        
        num_patterns = 2 * self.num_pattern_in_training + 1
        if self.state and self.get_state() != DEFAULT:
            num_patterns = 4 * self.num_pattern_in_training
            
        self.pattern_times = np.zeros(num_patterns)
        self.pattern_times[0] = ref_time
        
        for i in range(1, len(self.pattern_times)):
            if offset < len(p):
                self.pattern_times[i] = p[offset] + ref_time
                offset += 1
        
        # Recover historical scale/offset records
        self.retrieve_data_point_array(p, offset, SOCOEF)

    def get_model_param_dim(self) -> int:
        """
        Calculates the total dimension required for serializing this model.

        Returns:
            int: Total parameter count.
        """
        if self.state is None or self.get_state() == DEFAULT:
            return 6 * self.num_pattern_in_training + 2 + self.param_dim
        else:
            return 8 * self.num_pattern_in_training + 1 + self.param_dim

    def get_data_model_time(self, time: float) -> float:
        """
        Normalizes an absolute timestamp to a relative time feature.
        
        For state-based models, it calculates the time since the start of 
        the specific operational cycle.

        Args:
            time (float): The Unix timestamp.

        Returns:
            float: The normalized input value for the neural network.
        """
        if self.state is None or self.get_state() == DEFAULT:
            return super().get_data_model_time(time)
        else:
            num_pattern = len(self.pattern_times) // 2
            time_since = time - self.pattern_times[0]
            
            if time_since < 0:
                while time_since < 0:
                    time_since += self.pattern_period
            elif time > self.pattern_times[-1]:
                time_since = time - self.pattern_times[2 * (num_pattern - 1)]
                if time_since > self.pattern_period:
                    num = int(time_since / self.pattern_period)
                    time_since -= num * self.pattern_period
            else:
                for i in range(1, num_pattern):
                    if self.pattern_times[2*i] <= time < self.pattern_times[2*i + 1]:
                        time_since = time - self.pattern_times[2*i]
                        break
            
            return time_since / self.time_scale
