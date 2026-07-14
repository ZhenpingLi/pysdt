import os
import sys
from typing import List, Optional

import numpy as np

from config import sdt_config
from config.sdt_constants import POLYTREND

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.data_trend import DataTrend
import training.data_buffer

# --- Constants ---
LONGTERM = 1
MEAN_INDEX = 2


class PolyTrend(DataTrend):
    """
    Predictive model implementation for polynomial trends.
    
    This model captures linear or higher-order drift in telemetry data. It 
    constructs a polynomial feature vector from the time dimension (and 
    potentially other variables) and calculates predictions using learned 
    polynomial coefficients.
    """

    def __init__(self, mnemonic_id: str):
        """
        Initializes a new PolyTrend object.

        Args:
            mnemonic_id (str): Unique identifier for the telemetry mnemonic.
        """
        super().__init__(mnemonic_id)
        
        self.m: int = 0
        self.s_index: int = 0
        self.index: int = 0
        self.alg_name = POLYTREND
        
        if self.algorithm:
            dim = self.algorithm.get_dimension()
            if len(dim) == 1:
                self.s_index = 0
                self.m = dim[0]
            elif len(dim) >= 2:
                self.s_index = dim[0]
                self.m = dim[1]
        
        self.param_dim = self.m
        self.index = 0
        
        if training.data_buffer.session_type == LONGTERM:
            self.index = -1

    def get_pattern_period(self) -> float:
        """Returns the nominal cycle duration."""
        return self.pattern_period

    def get_trend_value(self, time: List[float]) -> float:
        """
        Calculates the predicted trend value for a given timestamp.
        
        It transforms the input into a polynomial feature vector and performs 
        a dot product with the learned model coefficients.

        Args:
            time (List[float]): A vector containing the timestamp (and other features).

        Returns:
            float: The predicted telemetry value.
        """
        if self.params is not None and len(time) >= 2:
            # Construct feature vector [t^s, t^(s+1), ..., t^(s+m-1)]
            features = self.poly_function(time)
            
            # Weighted sum for prediction
            output_value = np.dot(features, self.params)
            return float(output_value)
            
        else:
            # Fallback to mean if parameters are not available
            if self.stat_list and self.stat_list[0]:
                return float(self.stat_list[0].data[MEAN_INDEX])
            return 0.0

    def get_model_params(self) -> List[float]:
        """
        Serializes the model state into a flat list.
        
        The first element is the standard deviation (sigma), followed by the 
        polynomial coefficients.

        Returns:
            List[float]: The serialized parameter array.
        """
        if self.params is None and self.stat_list and self.stat_list[0]:
            self.params = np.zeros(self.m)
            self.params[0] = self.stat_list[0].data[MEAN_INDEX]
            
        model_params = np.zeros(self.get_model_param_dim())
        model_params[0] = self.sigma
        
        if self.params is not None:
            length = min(len(self.params), len(model_params) - 1)
            model_params[1 : 1 + length] = self.params[:length]
            
        return model_params.tolist()

    def set_model_params(self, p: List[float], ref_time: float):
        """
        Reconstructs the model from archived parameters and defines cycle context.
        
        It sets up a 48-hour session window split into two 24h pattern cycles.

        Args:
            p (List[float]): Source parameter array.
            ref_time (float): Reference start timestamp.
        """
        super().set_model_params(p, ref_time)
        
        session_period = sdt_config.session_period
        self.pattern_times = np.array([
            ref_time,
            ref_time + session_period / 2.0,
            ref_time + session_period
        ])
        self.pattern_period = session_period / 2.0

    def get_model_param_dim(self) -> int:
        """Returns the count of model coefficients plus the sigma element."""
        return self.param_dim + 1

    def poly_function(self, _input: List[float]) -> List[float]:
        """
        Generates the polynomial expansion terms for the feature vector.
        
        It creates a sequence of terms starting from degree 's_index' up to 
        'm-1' using the independent variable from the input vector.

        Args:
            _input (List[float]): Raw input vector [time, ...].

        Returns:
            List[float]: The expanded feature vector.
        """
        features = [0.0] * (self.m - self.s_index)
        features[0] = 1.0
        
        if len(_input) == 2:
            self.index = 0
            
        if self.s_index > 0:
            features[0] = np.power(_input[self.index + 1], self.s_index)
            
        for i in range(self.s_index + 1, self.m):
            features[i - self.s_index] = features[0] * _input[self.index + 1]

        return features
