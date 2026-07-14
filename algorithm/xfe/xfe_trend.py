from typing import List, Optional
import numpy as np
import sys
import os

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.data_trend import DataTrend
from algorithm.xfe.efe_function import EFEFunction

# --- Constants ---
HOUR_IN_SECONDS = 3600


class XFETrend(DataTrend):
    """
    Predictive model implementation for Extended Fourier Expansion (XFE).
    
    This model captures high-complexity periodic trends by extending the 
    standard Fourier series with additional harmonic terms. It uses a 
    dedicated EFEFunction for basis expansion and performs predictions based 
    on learned coefficients that are valid across an absolute time reference.
    """

    def __init__(self, mnemonic_id: str):
        """
        Initializes a new XFETrend object.

        Args:
            mnemonic_id (str): Unique identifier for the telemetry mnemonic.
        """
        super().__init__(mnemonic_id)
        
        self.dims: List[int] = [0, 0]
        self.func = EFEFunction()
        self.alg_name = "refea"
        
        if self.algorithm:
            self.dims = self.algorithm.get_dimension()
            # XFE parameter dimension: Bias + Multi-harmonic pairs
            order = self.dims[0] + self.dims[1]
            self.param_dim = 2 * order + 2
            self.params = np.zeros(self.param_dim)
            self.func.set_dim(self.dims)

    def set_order(self, new_dims: List[int]):
        """
        Redefines the harmonic orders for the expansion.

        Args:
            new_dims (List[int]): New dimensions [order1, order2].
        """
        if self.dims[0] != new_dims[0] or self.dims[1] != new_dims[1]:
            self.dims = new_dims
            self.func.set_dim(new_dims)
            order = self.dims[0] + self.dims[1]
            self.param_dim = 2 * order + 2
            self.params = np.zeros(self.param_dim)

    def set_pattern_period(self, p_p: float):
        """Configures the nominal cycle duration."""
        super().set_pattern_period(p_p)
        self.func.set_pattern_period(p_p)

    def get_trend_value(self, time: List[float]) -> float:
        """
        Calculates the predicted trend value for a given timestamp.
        
        Note: Unlike standard Fourier models, XFE uses time relative to 
        the absolute reference epoch, which allows it to model drift 
        across multiple pattern cycles.

        Args:
            time (List[float]): Input vector where time[0] is the timestamp.

        Returns:
            float: The predicted telemetry value.
        """
        # Normalized time in hours since the reference epoch
        time_since = (time[0] - self.ref_time) / HOUR_IN_SECONDS
        return self.func.get_trend(self.params, time_since)

    def get_model_params(self) -> List[float]:
        """
        Serializes the model's coefficients and metadata into a flat list.

        Returns:
            List[float]: The serialized parameter array including the period.
        """
        model_params = super().get_model_params()
        offset = self.param_dim + 1
        
        if offset < len(model_params):
            model_params[offset] = self.pattern_period
            
        return model_params

    def set_model_params(self, p: List[float], ref_time: float):
        """
        Reconstructs the model from archived parameters and context.

        Args:
            p (List[float]): Flat array from the archive.
            ref_time (float): Reference start timestamp.
        """
        super().set_model_params(p, ref_time)
        offset = self.param_dim + 1
        if offset < len(p):
            self.pattern_period = p[offset]

    def get_model_param_dim(self) -> int:
        """
        Returns the total dimension of the serialized model representation.

        Returns:
            int: The parameter count (coefficients + metadata).
        """
        return self.param_dim + 2
