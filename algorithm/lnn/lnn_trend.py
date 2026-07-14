from typing import List, Optional
import numpy as np
import sys
import os

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.data_trend import DataTrend
from algorithm.lnn.liquid_neural_network import LiquidNeuralNetwork

# Constants
HOUR_IN_SECONDS = 3600

class LNNTrend(DataTrend):
    """
    Represents a data trend model based on a Liquid Neural Network (LNN).
    """

    def __init__(self, mnemonic_id: str):
        super().__init__(mnemonic_id)
        self.alg_name="lnn"
        self.lnn: Optional[LiquidNeuralNetwork] = None
        self._init_network()

    def _init_network(self):
        """
        Initializes the Liquid Neural Network with a structure based on the algorithm's configuration.
        """
        if not self.algorithm:
            return
            
        dims = self.algorithm.get_dimension()
        if not dims or len(dims) < 1:
            # Default structure if not specified
            hidden_size = 10
        else:
            hidden_size = dims[0]

        # For a simple time-series trend, input and output size are typically 1
        input_size = 1
        output_size = 1
        
        # Time step is a crucial hyperparameter for LTC models
        time_step = float(self.algorithm.get_attribute("timestep") or 0.1)

        self.lnn = LiquidNeuralNetwork(input_size, hidden_size, output_size, time_step)
        
        # The total number of trainable parameters
        self.param_dim = self.lnn.lnn_dim

    def get_trend_value(self, time: List[float]) -> float:
        """
        Calculates the trend value for a given time using the LNN model.
        
        Note: For a single time point, this performs one forward step. The LNN's
        internal state is recurrent, so sequential calls are not independent.
        """
        if self.params is None or self.lnn is None:
            # Fallback to statistical mean if not trained
            stat = self.get_stat_at_time(time[0])
            return float(stat.data[2]) if stat else 0.0

        # Ensure the model has the correct weights
        self.lnn.set_weights_flat(np.array(self.params))
        
        # The input to the network is the normalized model time
        model_time = self.get_data_model_time(time[0])
        input_vec = np.array([model_time])
        
        # Perform a single forward pass
        output_vec = self.lnn.forward_step(input_vec)
        
        # Denormalize the output
        final_value = self.get_output(time[0], output_vec[0])
        
        return float(final_value)

    def get_output(self, time: float, orig_value: float) -> float:
        """
        Scales and offsets the raw output from the neural network.
        """
        scale_offset = self.get_scale_offset(time)
        if scale_offset is None or len(scale_offset) < 2:
            return orig_value
            
        scale = scale_offset[0] # SCALE index
        offset = scale_offset[1] # OFFSET index
        
        return orig_value / scale + offset if scale != 0 else offset

    def set_model_params(self, p: List[float], ref_time: float):
        """
        Sets the model parameters from a flat list, which includes sigma and all network weights/biases.
        """
        # The base class handles sigma and sets self.params
        super().set_model_params(p, ref_time)
        
        if self.lnn and self.params:
            self.lnn.set_weights_flat(np.array(self.params))
            
        # LNN-specific parameters (if any) would be deserialized here
        offset = self.param_dim + 1
        if offset < len(p):
            self.pattern_period = p[offset]

    def get_model_params(self) -> List[float]:
        """
        Exports the data model (sigma and all network weights/biases) to a flat list.
        """
        if self.lnn:
            # Ensure self.params is up-to-date with the network's state
            self.params = self.lnn.get_weights_flat().tolist()
        
        # The base class handles packaging sigma and self.params
        return super().get_model_params()

    def get_model_param_dim(self) -> int:
        """
        Gets the total dimension of the exportable model parameters.
        """
        # 1 (for sigma) + network params + 1 (for pattern_period)
        return self.param_dim + 2
