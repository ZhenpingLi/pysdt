import os
import sys
from typing import List

import numpy as np

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.ridgefe.ridge_fe_trend import RidgeFeTrend
from config.sdt_constants import DAY_IN_SECONDS, SHORTTERM, MEAN_INDEX
from algorithm.rmfe import fitting_function as ff


class MFETrend(RidgeFeTrend):
    """
    Predictive model implementation for Modified Fourier Expansion (MFE).
    
    MFETrend specializes the RidgeFeTrend for high-dimensional periodic data 
    (e.g., daily satellite patterns). It uses a custom fitting function to 
    map multi-variable inputs into a combined Fourier series, allowing for 
    more complex periodic behaviors than the standard RidgeFE.
    """

    def __init__(self, mnemonic_id: str):
        """
        Initializes a new MFETrend object.

        Args:
            mnemonic_id (str): Unique identifier for the telemetry mnemonic.
        """
        super().__init__(mnemonic_id)
        # Default time scale for MFE is normalized to a solar day
        self.time_scale = DAY_IN_SECONDS
        self.alg_name = "rmfe"
        self.func = ff
        # Register the parameter dimension with the fitting function
        self.func.set_dim(self.param_dim)

    def get_data_model_time(self, time: float) -> float:
        """
        Normalizes absolute time into cycle-relative hours [0, 24].
        
        Ensures the provided timestamp is wrapped correctly into a single 
        pattern period relative to the reference start time.

        Args:
            time (float): The Unix timestamp to normalize.

        Returns:
            float: Hours since the start of the current cycle.
        """
        if self.pattern_period == 0:
            return 0.0
            
        time_since = time - self.ref_time
        
        # Handle wraparound for periods
        if time_since < 0:
            time_since += self.pattern_period
        elif time_since > self.pattern_period:
            time_since -= self.pattern_period
        
        # Scale to days (then usually hours in the func implementation)
        return time_since / DAY_IN_SECONDS

    def set_pattern_period(self, p_p: float):
        """
        Configures the nominal cycle duration.

        Args:
            p_p (float): The period in seconds.
        """
        super().set_pattern_period(p_p)
        self.func.set_period(p_p)

    def set_data_model_time(self, p_times: np.ndarray, pattern_period: float):
        """
        Initializes the temporal context for training.

        Args:
            p_times (np.ndarray): Pattern boundary timestamps.
            pattern_period (float): Cycle duration in seconds.
        """
        self.set_pattern_period(pattern_period)
        self.set_pattern_times(p_times)
        self.num_pattern_in_training = 1

    def get_pattern_period(self) -> float:
        """Returns the current pattern period in seconds."""
        return self.pattern_period

    def set_model_params(self, p: List[float], ref_time: float):
        """
        Reconstructs the full MFE model state from a serialized parameter list.

        Args:
            p (List[float]): The flat array of coefficients and context.
            ref_time (float): The reference start timestamp.
        """
        super().set_model_params(p, ref_time)
        
        # Extract pattern period from the archived format
        offset = self.param_dim + 1
        if offset < len(p):
            self.set_pattern_period(p[offset])
            
        self.num_pattern_in_training = 1
        
        # Initialize a default 3-cycle window for evaluation
        self.pattern_times = ref_time + np.array([0, self.pattern_period, 2 * self.pattern_period])

    def get_trend_value(self, time: List[float]) -> float:
        """
        Calculates the predicted MFE trend value for a given timestamp.

        Args:
            time (List[float]): Input vector where time[0] is the timestamp.

        Returns:
            float: The predicted telemetry value.
        """
        if self.params is None:
             return 0.0
             
        _params = np.array(self.params)
        if self.training_type == SHORTTERM:
            sigma_t = self.get_sigma_t()
            # If successfully trended, perform Fourier inference
            if sigma_t > 0 and sigma_t != float('inf'):
                rel_time = self.get_data_model_time(time[0])
                return self.func.get_values(rel_time, _params)
            else:
                # Fallback to mean if no trend is available
                stat = self.get_stat_at_time(time[0])
                return float(stat.data[MEAN_INDEX]) if stat else 0.0
        else:
            # Long-term prediction
            rel_time = self.get_data_model_time(time[0])
            return self.func.get_values(rel_time, _params)
