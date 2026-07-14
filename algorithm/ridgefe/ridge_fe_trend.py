import math
import os
import sys
from typing import List, Optional

import numpy as np

from algorithm.algorithm_def import SHORTTERM
from config.sdt_constants import MEAN_INDEX
from training import data_buffer

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.data_trend import DataTrend
from algorithm.ridgefe.fourier_expansion_function import FourierExpansionFunction

# --- Constants ---
SCALE = 0
OFFSET = 1


class RidgeFeTrend(DataTrend):
    """
    Predictive model implementation for Ridge Fourier Expansion (RidgeFE).
    
    This class manages a FourierExpansionFunction to model periodic telemetry 
    trends. It maps learned Fourier coefficients to a time-varying signal 
    within orbital cycles. It automatically handles the dimensionality of 
    the Fourier expansion based on the algorithm's order and state definitions.
    """

    def __init__(self, mnemonic_id: str):
        """
        Initializes a new RidgeFeTrend object.

        Args:
            mnemonic_id (str): Unique identifier for the telemetry mnemonic.
        """
        super().__init__(mnemonic_id)
        
        self.order: int = 0
        self.func = FourierExpansionFunction()
        self.alg_name = "ridgefe"
        self._initialize_structure()

    def _initialize_structure(self):
        """
        Calculates the parameter dimensionality based on the Fourier expansion order.
        
        The order is retrieved from the algorithm definition. For multi-state 
        models, the order may be specific to the current state via 'dimpointer'.
        """
        if not self.algorithm:
            return
            
        dims = self.algorithm.get_dimension()
        if not dims:
            return
            
        if len(dims) == 1 or self.algorithm.al_type.state is None:
            self.order = dims[0]
        else:
            # Multi-state case: use dim_pointer if available
            if self.state is not None and len(dims) > 1:
                dim_pointer = self.state.dim_pointer
                if dim_pointer is not None:
                    self.order = dims[int(dim_pointer)]
                else:
                    self.order = dims[0]
            else:
                self.order = dims[0]

        # Parameter count = DC term + 2 * order (sine and cosine pairs)
        self.param_dim = 2 * self.order + 1
        self.func.set_dim([self.param_dim])

    def get_param_dim(self) -> int:
        """Returns the total number of Fourier coefficients."""
        return self.param_dim

    def set_data_model_time(self, pattern_times: np.ndarray, pattern_period: float):
        """
        Configures the temporal context and number of training patterns.

        Args:
            pattern_times (np.ndarray): Timestamps of pattern boundaries.
            pattern_period (float): The duration of a single cycle in seconds.
        """
        super().set_data_model_time(pattern_times, pattern_period)
        self.func.set_pattern_period(pattern_period)
        if pattern_times is not None and len(pattern_times) > 1:
            # RidgeFE typically uses a window of several pattern periods
            self.num_pattern_in_training = (len(pattern_times) - 1) // 2

    def set_pattern_period(self, p_p: float):
        """Overrides the nominal pattern period."""
        super().set_pattern_period(p_p)
        self.func.set_pattern_period(p_p)

    def get_trend_value(self, time: List[float]) -> float:
        """
        Calculates the predicted trend value for a given timestamp.
        
        It transforms the absolute time to cycle-relative time and evaluates 
        the learned Fourier series.

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
            # Check if the model was successfully trended
            if sigma_t > 0 and sigma_t != float('inf'):
                rel_time = self.get_data_model_time(time[0])
                return self.func.get_trend(_params, rel_time)
            else:
                # Fallback to mean for un-trended or static data
                stat = self.get_stat_at_time(time[0])
                return float(stat.data[MEAN_INDEX]) if stat else 0.0
        else: 
            # Long-term prediction
            return self.func.get_trend(_params, self.get_data_model_time(time[0]))

    def get_data_model_time(self, time: float) -> float:
        """
        Maps absolute Unix time to normalized hours since the current cycle start.

        Args:
            time (float): Absolute timestamp.

        Returns:
            float: Normalized relative time value.
        """
        if self.pattern_times is None or len(self.pattern_times) == 0 or self.pattern_period == 0:
            return 0.0
            
        # Identify the start of the current pattern cycle
        time_index = np.searchsorted(self.pattern_times, time, side='right') - 1
        time_since = time - self.pattern_times[time_index]
        
        # Periodic wrapping
        if time_since < 0:
            time_since += self.pattern_period
        if time_since > self.pattern_period:
            time_since -= self.pattern_period
            
        return time_since / self.time_scale

    def set_model_params(self, p: List[float], ref_time: float):
        """
        Deserializes model parameters from the archive.

        Args:
            p (List[float]): Flat array of coefficients and context metadata.
            ref_time (float): Reference start timestamp.
        """
        super().set_model_params(p, ref_time)
        default_model_time = data_buffer.get_default_model_time(self.algorithm)
        
        if default_model_time:
            p_period = default_model_time.get_model_period()
            self.set_pattern_period(p_period)
            
            num_patterns = 2 * self.num_pattern_in_training
            self.pattern_times = ref_time + np.arange(num_patterns) * self.pattern_period
