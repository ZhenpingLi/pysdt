from typing import List, Optional
import numpy as np
import sys
import os

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.data_trend import DataTrend
from training import data_buffer
from config.sdt_constants import LONGTERM

class TimePolyTrend(DataTrend):
    """
    Predictive model implementation for time-based polynomial trends.
    
    This model captures linear or higher-order drift in telemetry data as a 
    direct function of absolute time. It constructs a polynomial feature 
    vector from the time input and calculates predictions using learned 
    polynomial coefficients.
    """

    def __init__(self, mnemonic_id: str):
        """
        Initializes a new TimePolyTrend object.

        Args:
            mnemonic_id (str): Unique identifier for the telemetry mnemonic.
        """
        super().__init__(mnemonic_id)
        
        self.m: int = 0
        self.s_index: int = 0
        self.index: int = 0
        self.alg_name = "timepoly"
        self._initialize_structure()

    def _initialize_structure(self):
        """
        Configures the polynomial degree (m) and starting index (s_index) 
        from the algorithm's dimension attributes.
        """
        if not self.algorithm:
            return
            
        dim = self.algorithm.get_dimension()
        if len(dim) == 1:
            self.s_index = 0
            self.m = dim[0]
        elif len(dim) >= 2:
            self.s_index = dim[0]
            self.m = dim[1]
        
        self.param_dim = self.m
        
        if data_buffer.session_type == LONGTERM:
            self.index = -1

    def get_pattern_period(self) -> float:
        """
        Returns the effective pattern period for this trend.
        
        For TimePoly, this is typically a multiple of the base pattern period 
        to cover a broader time range.

        Returns:
            float: The pattern period in seconds.
        """
        return 14 * self.pattern_period

    def get_trend_value(self, time: List[float]) -> float:
        """
        Calculates the predicted trend value for a given timestamp.
        
        It transforms the absolute time into a polynomial feature vector 
        (relative to the model's reference time) and performs a dot product 
        with the learned model coefficients.

        Args:
            time (List[float]): A vector containing the timestamp as the first element.

        Returns:
            float: The predicted telemetry value.
        """
        input_time_relative = time[0] - self.ref_time
        
        if self.params is None:
            return 0.0
            
        params = np.array(self.params)
        
        # Create array of powers: [s_index, s_index+1, ..., s_index+m-1]
        powers = np.arange(self.s_index, self.s_index + self.m)
        
        # Calculate terms: input_time_relative^power
        terms = np.power(input_time_relative, powers)
        
        # Perform dot product for prediction: y = sum(params[i] * terms[i])
        value = np.sum(params * terms)
        
        return float(value)

    def set_model_params(self, p: List[float], ref_time: float):
        """
        Reconstructs the model from archived parameters and defines cycle context.
        
        It sets up a 48-hour session window split into two 24h pattern cycles.

        Args:
            p (List[float]): Source parameter array.
            ref_time (float): Reference start timestamp.
        """
        super().set_model_params(p, ref_time)
        
        session_period = data_buffer.session_period * 3600 # Convert hours to seconds
        
        self.pattern_times = np.array([
            ref_time,
            ref_time + session_period / 2.0,
            ref_time + session_period
        ])
        self.pattern_period = session_period / 2.0

    def set_pattern_times(self, pattern_times: np.ndarray):
        """
        Sets the pattern times and updates the pattern period and reference time.

        Args:
            pattern_times (np.ndarray): Array of pattern boundary timestamps.
        """
        self.pattern_times = pattern_times
        if pattern_times is not None and len(pattern_times) > 1:
            self.pattern_period = pattern_times[1] - pattern_times[0]
            self.ref_time = pattern_times[0]

    def get_model_param_dim(self) -> int:
        """
        Returns the total dimension of the model's parameter array.
        
        This includes the standard deviation (sigma) and all polynomial coefficients.

        Returns:
            int: The parameter count (param_dim + 1).
        """
        return self.param_dim + 1

    def get_model_params(self) -> List[float]:
        """
        Serializes the model state into a flat list.
        
        The first element is the standard deviation (sigma), followed by the 
        polynomial coefficients.

        Returns:
            List[float]: The serialized parameter array.
        """
        s_array = [0.0] * self.get_model_param_dim()
        s_array[0] = self.sigma
        
        if self.params is not None:
            # Copy polynomial coefficients into the array
            s_array[1 : self.param_dim + 1] = self.params.tolist()
        else:
            # Default to zeros if no parameters are set
            s_array[1 : self.param_dim + 1] = [0.0] * self.param_dim

        return s_array
